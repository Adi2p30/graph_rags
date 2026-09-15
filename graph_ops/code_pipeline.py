"""Code-graph ingestion: walk a source tree -> tree-sitter chunk every file
(ingestion/code_chunker.py) -> resolve calls/imports/inheritance into edges
by reasoning across the *whole* repo's symbol table -> embed + write to
Neo4j/Qdrant. Mirrors graph_ops/pipeline.py's shape (phase 1: structure,
phase 2: embed + store) with one extra phase in between: call/import/
inheritance resolution needs every file's symbols and import table in hand
at once, which a single-file chunker can't do on its own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from graphbuild.embeddings import embed_texts
from graphbuild.node_schema import (
    CodeFileNode, CodeSymbolNode, EdgeMethod, GraphEdge, RelationType, make_edge, new_id,
)
from graphbuild.sparse import SparseIndex, SparseVector
from ingestion.code_chunker import (
    FileChunkResult, build_embedding_text, build_sparse_text, chunk_code_file, language_for_path,
)
from store import neo4j_store, qdrant_store


@dataclass
class CodeIngestReport:
    files: list[dict] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    total_symbols: int = 0
    edges_created: dict[str, int] = field(default_factory=dict)
    sparse_index_mode: str = ""
    elapsed_seconds: float = 0.0


def _module_path_for(rel_path: Path) -> str:
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_source_files(repo_root: Path) -> list[Path]:
    out = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in config.CODE_EXCLUDE_DIRS for part in path.relative_to(repo_root).parts[:-1]):
            continue
        if path.suffix not in config.CODE_SUPPORTED_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > config.CODE_MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(path)
    return out


def _load_and_chunk_file(repo_root: Path, path: Path, repo: str, topic: str) -> FileChunkResult | None:
    rel_path = path.relative_to(repo_root)
    language = language_for_path(str(path))
    if language is None:
        return None
    try:
        source_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    file_node = CodeFileNode(
        id=new_id("file"),
        type=None,
        label=str(rel_path),
        topic=topic,
        repo=repo,
        path=str(rel_path),
        module_path=_module_path_for(rel_path),
        language=language,
        loc=source_text.count("\n") + 1,
    )
    try:
        return chunk_code_file(source_text, file_node)
    except Exception:
        # tree-sitter itself is error-tolerant (it emits ERROR nodes rather
        # than raising), but our extraction walk assumes a shape the grammar
        # guarantees -- if an edge case still slips through, one malformed
        # file shouldn't sink a whole-repo batch ingest. Fall back to a
        # single raw "module" symbol so the file is still represented.
        symbol = CodeSymbolNode(
            id=new_id("sym"), type=None, label=f"{file_node.path} (unparsed)",
            topic=file_node.topic, file_id=file_node.id, kind="module", language=file_node.language,
            name=file_node.module_path, qualified_name=file_node.module_path,
            text=source_text[:config.CODE_CHUNK_MAX_BODY_CHARS],
            truncated=len(source_text) > config.CODE_CHUNK_MAX_BODY_CHARS,
            start_line=1, end_line=file_node.loc, order_index=0, created_by="pipeline",
        )
        return FileChunkResult(file=file_node, symbols=[symbol], import_table={})


# --- phase 2: repo-wide call/import/inheritance resolution ------------------


def _build_indices(results: list[FileChunkResult]) -> tuple[
    dict[str, CodeSymbolNode], dict[str, list[CodeSymbolNode]], dict[str, CodeSymbolNode], dict[str, CodeFileNode],
]:
    qualified_index: dict[str, CodeSymbolNode] = {}
    name_index: dict[str, list[CodeSymbolNode]] = {}
    symbol_by_id: dict[str, CodeSymbolNode] = {}
    module_index: dict[str, CodeFileNode] = {}

    for result in results:
        module_index[result.file.module_path] = result.file
        for sym in result.symbols:
            symbol_by_id[sym.id] = sym
            if sym.kind == "module":
                continue
            qualified_index[sym.qualified_name] = sym
            name_index.setdefault(sym.name, []).append(sym)

    return qualified_index, name_index, symbol_by_id, module_index


def _resolve_name(
    raw: str, symbol: CodeSymbolNode, import_table: dict[str, str],
    qualified_index: dict[str, CodeSymbolNode], name_index: dict[str, list[CodeSymbolNode]],
    symbol_by_id: dict[str, CodeSymbolNode], kind_filter: str | None = None,
) -> tuple[CodeSymbolNode | None, float]:
    """Best-effort resolution of a raw AST-extracted name (a call's callee
    expression, or a class's raw base-class expression) to a CodeSymbolNode
    already in this repo's symbol table. Confidence (edge weight) reflects
    how the match was made: exact self/cls or import-qualified resolution is
    1.0; an unambiguous same-file or global name match is a lower-confidence
    heuristic. Deliberately returns None rather than guessing when a bare
    name matches more than one candidate -- a wrong edge is worse than a
    missing one."""
    raw = raw.strip()
    if not raw:
        return None, 0.0
    parts = raw.split(".")
    root, rest = parts[0], parts[1:]

    def _filtered(candidates: list[CodeSymbolNode]) -> list[CodeSymbolNode]:
        if kind_filter is None:
            return candidates
        return [c for c in candidates if c.kind == kind_filter]

    if root in ("self", "cls") and symbol.parent_symbol_id and len(rest) == 1:
        parent = symbol_by_id.get(symbol.parent_symbol_id)
        if parent and parent.kind == "class":
            target = qualified_index.get(f"{parent.qualified_name}.{rest[0]}")
            if target and target in _filtered([target]):
                return target, 1.0

    if root in import_table:
        candidate = import_table[root]
        if rest:
            target = qualified_index.get(f"{candidate}.{'.'.join(rest)}")
            if target and target in _filtered([target]):
                return target, 1.0
        target = qualified_index.get(candidate)
        if target and target in _filtered([target]):
            return target, 1.0

    if not rest:
        # Bare identifier call/reference (no attribute access) -- root is
        # either a local function/class name or a builtin, so a same-file or
        # unique-repo-wide match on that exact name is a reasonable guess.
        candidates = _filtered(name_index.get(root, []))
        same_file = [c for c in candidates if c.file_id == symbol.file_id]
        if len(same_file) == 1:
            return same_file[0], 0.9
        if len(candidates) == 1:
            return candidates[0], 0.4
        return None, 0.0

    # A dotted call/reference through a receiver we couldn't resolve via
    # self/cls or the import table (e.g. `some_local_var.get(...)`) has no
    # type information behind it -- guessing by the trailing attribute name
    # alone produces false edges for common method names (`.get`, `.run`,
    # `.close`, ...) that collide with an unrelated same-named repo function.
    # Deliberately unresolved rather than guessed.
    return None, 0.0


def _resolve_module(candidate: str, module_index: dict[str, CodeFileNode]) -> CodeFileNode | None:
    segments = candidate.split(".")
    while segments:
        hit = module_index.get(".".join(segments))
        if hit:
            return hit
        segments = segments[:-1]
    return None


def resolve_repo_edges(results: list[FileChunkResult]) -> list[GraphEdge]:
    qualified_index, name_index, symbol_by_id, module_index = _build_indices(results)
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(source_id: str, target_id: str, rel_type: RelationType, weight: float, method: EdgeMethod) -> None:
        key = (source_id, target_id, rel_type.value)
        if key in seen or source_id == target_id and rel_type != RelationType.CALLS:
            return
        seen.add(key)
        edges.append(make_edge(source_id, target_id, rel_type, method, weight=weight))

    for result in results:
        file = result.file
        import_table = result.import_table

        target_files = {c: _resolve_module(c, module_index) for c in import_table.values()}
        for target_file in {f.id: f for f in target_files.values() if f}.values():
            if target_file.id != file.id:
                _add(file.id, target_file.id, RelationType.IMPORTS, 1.0, EdgeMethod.STATIC_ANALYSIS)

        for sym in result.symbols:
            if sym.kind == "class":
                for raw_base in sym.base_classes:
                    target, weight = _resolve_name(
                        raw_base, sym, import_table, qualified_index, name_index, symbol_by_id, kind_filter="class",
                    )
                    if target:
                        _add(sym.id, target.id, RelationType.INHERITS_FROM, weight, EdgeMethod.STATIC_ANALYSIS)

            if sym.kind in ("function", "method"):
                for raw_call in sym.calls_raw:
                    target, weight = _resolve_name(
                        raw_call, sym, import_table, qualified_index, name_index, symbol_by_id,
                    )
                    if target:
                        _add(sym.id, target.id, RelationType.CALLS, weight, EdgeMethod.STATIC_ANALYSIS)

    return edges


def _structural_edges(result: FileChunkResult) -> list[GraphEdge]:
    edges = []
    for sym in result.symbols:
        parent_id = sym.parent_symbol_id or result.file.id
        edges.append(make_edge(parent_id, sym.id, RelationType.HAS_CHUNK, EdgeMethod.STRUCTURAL))
        if sym.next_chunk_id:
            edges.append(make_edge(sym.id, sym.next_chunk_id, RelationType.NEXT_CHUNK, EdgeMethod.STRUCTURAL))
    return edges


# --- orchestration -------------------------------------------------------


def run_code_ingest(
    repo_root: str | Path,
    repo: str,
    topic: str = "code",
    run_connection_finder: bool = False,
) -> CodeIngestReport:
    start = time.time()
    report = CodeIngestReport()
    repo_root = Path(repo_root).resolve()
    neo4j_store.init_schema()

    results: list[FileChunkResult] = []
    for path in _iter_source_files(repo_root):
        result = _load_and_chunk_file(repo_root, path, repo, topic)
        if result is None:
            report.files_skipped.append(str(path.relative_to(repo_root)))
            continue
        results.append(result)
        report.files.append({
            "path": result.file.path, "symbols": len(result.symbols), "loc": result.file.loc,
        })

    if not results:
        report.elapsed_seconds = time.time() - start
        return report

    structural_edges: list[GraphEdge] = []
    for result in results:
        structural_edges.extend(_structural_edges(result))
    static_edges = resolve_repo_edges(results)

    all_files = [r.file for r in results]
    all_symbols = [sym for r in results for sym in r.symbols]
    report.total_symbols = len(all_symbols)

    neo4j_store.upsert_nodes(all_files)
    neo4j_store.upsert_nodes(all_symbols)
    neo4j_store.upsert_edges(structural_edges)
    neo4j_store.upsert_edges(static_edges)

    for edge in static_edges:
        report.edges_created[edge.rel_type.value] = report.edges_created.get(edge.rel_type.value, 0) + 1
    report.edges_created["HAS_CHUNK"] = sum(1 for e in structural_edges if e.rel_type == RelationType.HAS_CHUNK)
    report.edges_created["NEXT_CHUNK"] = sum(1 for e in structural_edges if e.rel_type == RelationType.NEXT_CHUNK)

    # Embedding text: a structured "card" (qualified name, signature, docstring,
    # callees) built by code_chunker, not raw source -- see that module's
    # docstring for why. TF-IDF is a corpus-wide shared resource (see
    # ingestion/chunker.py / pipeline.py): if a books index already exists we
    # extend it via transform() so the sparse vector space code chunks land in
    # stays consistent with prose chunks in the same Qdrant collection; only a
    # fresh install with no prior ingest fits+saves a new one.
    file_by_id = {f.id: f for f in all_files}
    embed_texts_in = [build_embedding_text(sym, file_by_id[sym.file_id]) for sym in all_symbols]
    sparse_texts_in = [build_sparse_text(sym, file_by_id[sym.file_id]) for sym in all_symbols]

    try:
        sparse_index = SparseIndex.load()
        sparse_matrix = sparse_index.transform(sparse_texts_in)
        report.sparse_index_mode = "extended_existing"
    except FileNotFoundError:
        sparse_index = SparseIndex()
        sparse_matrix = sparse_index.fit_transform(sparse_texts_in)
        sparse_index.save()
        report.sparse_index_mode = "fit_new"

    dense_vecs = embed_texts(embed_texts_in, show_progress=True)
    sparse_vecs = [
        SparseVector(indices=sparse_matrix.getrow(i).indices.tolist(), values=sparse_matrix.getrow(i).data.tolist())
        for i in range(len(sparse_texts_in))
    ]

    payloads = []
    for sym, embed_text in zip(all_symbols, embed_texts_in):
        payload = sym.to_dict()
        payload.pop("metadata", None)
        payload["text"] = embed_text  # what graphrag_answerer shows/cites is the enrichment card, not raw code
        payload["raw_code"] = sym.text
        payload["file_path"] = file_by_id[sym.file_id].path
        payloads.append(payload)

    qdrant_store.upsert_points(
        node_ids=[s.id for s in all_symbols],
        dense_vecs=[v.tolist() for v in dense_vecs],
        sparse_vecs=sparse_vecs,
        payloads=payloads,
    )

    if run_connection_finder:
        from graph_ops.edge_ops import run_connection_finder_pass
        report.edges_created["connection_finder_pass"] = run_connection_finder_pass(topic=topic)

    report.elapsed_seconds = time.time() - start
    return report
