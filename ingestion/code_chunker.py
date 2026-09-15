"""Structure-aware code chunking: tree-sitter (https://tree-sitter.github.io/tree-sitter/)
parses each file into a concrete syntax tree, which we walk to find *semantic*
units -- module, class, function, method -- rather than cutting at arbitrary
line/token windows. This is what "graph structure based on the structure of
the code" means concretely: CodeFile -HAS_CHUNK-> symbol (top-level class or
function), class-symbol -HAS_CHUNK-> method symbol, plus symbol -NEXT_CHUNK->
symbol for definition order -- the same shape ingestion/chunker.py uses for
Book -HAS_SECTION-> Section -HAS_CHUNK-> Chunk.

Tree-sitter alone only gives you a parse tree; it doesn't know what a
"qualified name" or a "call graph" is. The structured-extraction layer on top
turns grammar nodes into graph-meaningful facts per symbol: a resolved
qualified name, a rendered signature, docstring, decorators, superclasses, a
cyclomatic-complexity proxy (branch-node count), and the raw callee
expressions found in its body. Those raw callees/superclasses/imports are
*not* resolved into edges here -- that requires seeing every file in the repo
at once, so it's a second pass in graph_ops/code_pipeline.py. What this module
guarantees is that the raw material for that pass (qualified_name,
import_table, calls_raw, base_classes) is captured accurately from the AST
instead of regexed out of text.

Adding a language means adding one entry to _LANGUAGE_LOADERS/LANGUAGE_BY_EXT
plus a LangSpec -- the walk/extract functions below are all driven off that
table, not per-language branches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import config
from graphbuild.node_schema import CodeFileNode, CodeSymbolNode, new_id

from tree_sitter import Language, Node, Parser

# --- language registry ------------------------------------------------------


@dataclass(frozen=True)
class LangSpec:
    name: str
    class_node_types: frozenset[str]
    function_node_types: frozenset[str]
    decorated_wrapper: Optional[str]     # node type wrapping decorators + a definition, or None
    call_node_type: str
    branch_node_types: frozenset[str]    # counted for the complexity proxy
    import_node_types: frozenset[str]
    string_node_type: str


def _load_python() -> tuple[Language, LangSpec]:
    import tree_sitter_python as tspython

    lang = Language(tspython.language())
    spec = LangSpec(
        name="python",
        class_node_types=frozenset({"class_definition"}),
        function_node_types=frozenset({"function_definition"}),
        decorated_wrapper="decorated_definition",
        call_node_type="call",
        branch_node_types=frozenset({
            "if_statement", "elif_clause", "for_statement", "while_statement",
            "except_clause", "with_statement", "boolean_operator",
            "conditional_expression", "match_statement", "case_clause",
        }),
        import_node_types=frozenset({"import_statement", "import_from_statement", "future_import_statement"}),
        string_node_type="string",
    )
    return lang, spec


_LANGUAGE_LOADERS = {"python": _load_python}
LANGUAGE_BY_EXT = {".py": "python"}


@lru_cache(maxsize=None)
def _get_language_and_spec(language: str) -> tuple[Language, LangSpec]:
    loader = _LANGUAGE_LOADERS.get(language)
    if loader is None:
        raise ValueError(f"code_chunker: unsupported language {language!r}")
    return loader()


@lru_cache(maxsize=None)
def _get_parser(language: str) -> Parser:
    lang, _ = _get_language_and_spec(language)
    return Parser(lang)


def language_for_path(path: str) -> Optional[str]:
    for ext, lang in LANGUAGE_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return None


# --- small AST helpers -------------------------------------------------------


def _text(node: Node, source: bytes) -> str:
    # tree-sitter offsets are byte offsets, not char offsets -- must slice
    # the encoded source, never the decoded str.
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _line(node: Node) -> int:
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    return node.end_point[0] + 1


_STRING_PREFIX_RE = re.compile(r'^[a-zA-Z]{0,3}')
_QUOTE_RE = re.compile(r'^("""|\'\'\'|"|\')')


def _clean_docstring(raw: str) -> str:
    s = raw.strip()
    prefix_match = _STRING_PREFIX_RE.match(s)
    if prefix_match:
        s = s[prefix_match.end():]
    quote_match = _QUOTE_RE.match(s)
    if quote_match:
        q = quote_match.group(1)
        s = s[len(q):]
        if s.endswith(q):
            s = s[:-len(q)]
    return s.strip()


def _docstring_of(body: Optional[Node], spec: LangSpec, source: bytes) -> str:
    if body is None or body.child_count == 0:
        return ""
    first = body.children[0]
    if first.type != "expression_statement" or first.child_count == 0:
        return ""
    inner = first.children[0]
    if inner.type != spec.string_node_type:
        return ""
    return _clean_docstring(_text(inner, source))


def _collect_in_scope(node: Node, spec: LangSpec, predicate) -> list[Node]:
    """Depth-first collect of descendants matching `predicate`, stopping at
    (not descending into) nested class/function definitions -- those become
    their own symbols with their own calls/complexity, so a method's calls
    shouldn't be double-attributed to its enclosing class, etc."""
    out: list[Node] = []
    for child in node.children:
        if child.type == spec.decorated_wrapper or \
           child.type in spec.class_node_types or \
           child.type in spec.function_node_types:
            continue
        if predicate(child):
            out.append(child)
        out.extend(_collect_in_scope(child, spec, predicate))
    return out


def _callee_text(call_node: Node, source: bytes) -> Optional[str]:
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    return _text(fn, source)


def _extract_calls(def_node: Node, spec: LangSpec, source: bytes) -> list[str]:
    calls = _collect_in_scope(def_node, spec, lambda n: n.type == spec.call_node_type)
    out = []
    for c in calls:
        callee = _callee_text(c, source)
        if callee:
            out.append(callee)
    return out


def _branch_count(def_node: Node, spec: LangSpec) -> int:
    return len(_collect_in_scope(def_node, spec, lambda n: n.type in spec.branch_node_types))


def _base_classes(class_node: Node, source: bytes) -> list[str]:
    superclasses = class_node.child_by_field_name("superclasses")
    if superclasses is None:
        return []
    bases = []
    for child in superclasses.children:
        # argument_list children include the parens/commas plus each base
        # expression; keyword args like `metaclass=Meta` show up as
        # `keyword_argument` and are deliberately excluded -- they aren't
        # base classes.
        if child.type in ("identifier", "attribute"):
            bases.append(_text(child, source))
    return bases


def _decorators_of(decorated_node: Node, source: bytes) -> list[str]:
    return [
        _text(child, source) for child in decorated_node.children
        if child.type == "decorator"
    ]


def _signature_of(def_node: Node, source: bytes) -> str:
    params = def_node.child_by_field_name("parameters")
    ret = def_node.child_by_field_name("return_type")
    sig = _text(params, source) if params is not None else "()"
    if ret is not None:
        sig += f" -> {_text(ret, source)}"
    return sig


# --- import extraction -------------------------------------------------------


def _dotted_name_text(node: Node, source: bytes) -> str:
    return _text(node, source)


def _parse_import_statement(node: Node, source: bytes) -> list[tuple[str, str]]:
    """import a; import a.b as c  ->  [(local_name, module_path), ...]"""
    out = []
    for child in node.children:
        if child.type == "dotted_name":
            dotted = _dotted_name_text(child, source)
            local = dotted.split(".")[0]
            out.append((local, dotted))
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node is not None and alias_node is not None:
                out.append((_text(alias_node, source), _dotted_name_text(name_node, source)))
    return out


def _parse_import_from_statement(node: Node, source: bytes) -> list[tuple[str, str]]:
    """from pkg.mod import a, b as c  ->  [(a, "pkg.mod.a"), (c, "pkg.mod.b")]"""
    module_node = node.child_by_field_name("module_name")
    if module_node is None:
        return []
    module = _dotted_name_text(module_node, source)
    module_span = (module_node.start_byte, module_node.end_byte)
    out = []
    for child in node.children:
        # `is not module_node`: tree-sitter Node wrappers aren't guaranteed
        # identical objects across access paths, so compare byte spans.
        if child.type == "dotted_name" and (child.start_byte, child.end_byte) != module_span:
            name = _dotted_name_text(child, source)
            out.append((name, f"{module}.{name}"))
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node is not None and alias_node is not None:
                out.append((_text(alias_node, source), f"{module}.{_dotted_name_text(name_node, source)}"))
        elif child.type == "wildcard_import":
            out.append(("*", module))
    return out


def _extract_imports(root: Node, spec: LangSpec, source: bytes) -> tuple[list[str], dict[str, str]]:
    """Returns (raw import source lines, {local_name: candidate_qualified_path})."""
    raw: list[str] = []
    table: dict[str, str] = {}
    for child in root.children:
        if child.type not in spec.import_node_types:
            continue
        raw.append(_text(child, source))
        if child.type == "import_statement":
            pairs = _parse_import_statement(child, source)
        else:
            pairs = _parse_import_from_statement(child, source)
        for local, candidate in pairs:
            if local != "*":
                table[local] = candidate
    return raw, table


# --- symbol walk --------------------------------------------------------


@dataclass
class _WalkCtx:
    file: CodeFileNode
    spec: LangSpec
    source: bytes
    symbols: list[CodeSymbolNode] = field(default_factory=list)
    order_index: int = 0


def _make_symbol(
    ctx: _WalkCtx, kind: str, def_node: Node, name: str, qualified_name: str,
    parent_symbol_id: Optional[str], decorators: list[str],
) -> CodeSymbolNode:
    body = def_node.child_by_field_name("body")
    is_callable = kind in ("function", "method")
    raw_text = _text(def_node, ctx.source)
    truncated = len(raw_text) > config.CODE_CHUNK_MAX_BODY_CHARS
    symbol = CodeSymbolNode(
        id=new_id("sym"),
        type=None,
        label=qualified_name[:120],
        topic=ctx.file.topic,
        file_id=ctx.file.id,
        parent_symbol_id=parent_symbol_id,
        kind=kind,
        language=ctx.spec.name,
        name=name,
        qualified_name=qualified_name,
        signature=_signature_of(def_node, ctx.source) if is_callable else "",
        docstring=_docstring_of(body, ctx.spec, ctx.source) if kind in ("class", "function", "method") else "",
        decorators=decorators,
        base_classes=_base_classes(def_node, ctx.source) if kind == "class" else [],
        calls_raw=_extract_calls(def_node, ctx.spec, ctx.source) if is_callable else [],
        start_line=_line(def_node),
        end_line=_end_line(def_node),
        complexity=_branch_count(def_node, ctx.spec) if is_callable else 0,
        text=raw_text[:config.CODE_CHUNK_MAX_BODY_CHARS],
        truncated=truncated,
        order_index=ctx.order_index,
        created_by="pipeline",
    )
    ctx.order_index += 1
    ctx.symbols.append(symbol)
    return symbol


def _walk_body(ctx: _WalkCtx, container: Node, qualified_prefix: str, parent_symbol_id: Optional[str]) -> None:
    """Depth-first, left-to-right over `container`'s direct children, unwrapping
    decorated_definition, recursing into class/function bodies to find nested
    definitions in source (definition) order -- which is what NEXT_CHUNK/
    order_index end up encoding."""
    spec = ctx.spec
    for child in container.children:
        decorators: list[str] = []
        node = child
        if node.type == spec.decorated_wrapper:
            decorators = _decorators_of(node, ctx.source)
            node = node.child_by_field_name("definition")
            if node is None:
                continue

        if node.type in spec.class_node_types:
            name_node = node.child_by_field_name("name")
            name = _text(name_node, ctx.source) if name_node else "<anonymous>"
            qn = f"{qualified_prefix}.{name}" if qualified_prefix else name
            symbol = _make_symbol(ctx, "class", node, name, qn, parent_symbol_id, decorators)
            body = node.child_by_field_name("body")
            if body is not None:
                _walk_body(ctx, body, qn, symbol.id)

        elif node.type in spec.function_node_types:
            name_node = node.child_by_field_name("name")
            name = _text(name_node, ctx.source) if name_node else "<anonymous>"
            qn = f"{qualified_prefix}.{name}" if qualified_prefix else name
            kind = "method" if parent_symbol_id and ctx.symbols and \
                _find_symbol(ctx, parent_symbol_id).kind == "class" else "function"
            symbol = _make_symbol(ctx, kind, node, name, qn, parent_symbol_id, decorators)
            body = node.child_by_field_name("body")
            if body is not None:
                _walk_body(ctx, body, qn, symbol.id)


def _find_symbol(ctx: _WalkCtx, symbol_id: str) -> CodeSymbolNode:
    for s in ctx.symbols:
        if s.id == symbol_id:
            return s
    raise KeyError(symbol_id)


# --- module-level ("loose" top-level code) packing ---------------------------


def _pack_module_chunks(
    ctx: _WalkCtx, root: Node, module_docstring: str, imports_raw: list[str],
) -> list[CodeSymbolNode]:
    """Everything at file scope that ISN'T an import or a def/class -- module
    docstring, constants, `if __name__ == '__main__':`, script bodies -- gets
    packed into one or more "module" symbols bounded by
    config.CODE_MODULE_CHUNK_MAX_LINES, mirroring how ingestion/chunker.py
    packs prose paragraphs into word-budget chunks. Always emits at least one
    module chunk (even an empty file gets a stub) so every CodeFile has a
    retrievable file-level overview and a head for the NEXT_CHUNK chain."""
    spec = ctx.spec
    loose_spans: list[tuple[int, int, str]] = []  # (start_line, end_line, text)
    for idx, child in enumerate(root.children):
        if child.type in spec.import_node_types:
            continue
        if child.type == spec.decorated_wrapper or \
           child.type in spec.class_node_types or child.type in spec.function_node_types:
            continue
        if idx == 0 and module_docstring and child.type == "expression_statement":
            continue  # the module docstring itself -- already captured separately.
        loose_spans.append((_line(child), _end_line(child), _text(child, ctx.source)))

    header_lines = []
    if module_docstring:
        header_lines.append(module_docstring)
    if imports_raw:
        header_lines.append("Imports:\n" + "\n".join(imports_raw))
    header = "\n\n".join(header_lines)

    groups: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    current_lines = 0
    for span in loose_spans:
        span_lines = span[1] - span[0] + 1
        if current and current_lines + span_lines > config.CODE_MODULE_CHUNK_MAX_LINES:
            groups.append(current)
            current, current_lines = [], 0
        current.append(span)
        current_lines += span_lines
    if current:
        groups.append(current)
    if not groups:
        groups = [[]]

    chunks = []
    for i, group in enumerate(groups):
        body_text = "\n".join(s[2] for s in group)
        text = (header if i == 0 else "") + (("\n\n" if i == 0 and header else "") + body_text if body_text else "")
        if not text.strip():
            if i > 0:
                continue
            text = f"(empty module: {ctx.file.path})"
        start_line = group[0][0] if group else 1
        end_line = group[-1][1] if group else 1
        suffix = "" if i == 0 else f"#{i}"
        symbol = CodeSymbolNode(
            id=new_id("sym"),
            type=None,
            label=f"{ctx.file.path} (module){suffix}",
            topic=ctx.file.topic,
            file_id=ctx.file.id,
            parent_symbol_id=None,
            kind="module",
            language=spec.name,
            name=ctx.file.module_path.rsplit(".", 1)[-1] if ctx.file.module_path else ctx.file.path,
            qualified_name=f"{ctx.file.module_path}{suffix}" if ctx.file.module_path else f"{ctx.file.path}{suffix}",
            docstring=module_docstring,
            start_line=start_line,
            end_line=end_line,
            text=text[:config.CODE_CHUNK_MAX_BODY_CHARS],
            truncated=len(text) > config.CODE_CHUNK_MAX_BODY_CHARS,
            order_index=-1,  # fixed up (module chunk(s) always lead the file) by caller
            created_by="pipeline",
        )
        chunks.append(symbol)
    return chunks


# --- public API ---------------------------------------------------------


@dataclass
class FileChunkResult:
    file: CodeFileNode
    symbols: list[CodeSymbolNode]          # module chunk(s) first, then definitions in source order
    import_table: dict[str, str]           # local_name -> candidate qualified path, for CALLS/IMPORTS resolution


def chunk_code_file(source_text: str, file: CodeFileNode) -> FileChunkResult:
    """Parses one file's source with tree-sitter and returns its CodeFile's
    semantic chunks: a leading 'module' symbol/symbols (docstring + imports +
    loose top-level code), then every class/function/method found by walking
    the AST, each carrying its own structural + static-analysis-ready
    metadata. Never raises on malformed/partial source -- tree-sitter is
    error-tolerant by design and simply produces ERROR nodes we don't walk
    into, so worst case a broken file degrades to just its module chunk."""
    _, spec = _get_language_and_spec(file.language)
    parser = _get_parser(file.language)
    source = source_text.encode("utf-8", errors="replace")
    tree = parser.parse(source)
    root = tree.root_node

    imports_raw, import_table = _extract_imports(root, spec, source)
    module_docstring = _docstring_of(root, spec, source)

    ctx = _WalkCtx(file=file, spec=spec, source=source)
    module_chunks = _pack_module_chunks(ctx, root, module_docstring, imports_raw)
    _walk_body(ctx, root, file.module_path, parent_symbol_id=None)

    ordered = module_chunks + ctx.symbols
    for i, sym in enumerate(ordered):
        sym.order_index = i

    # NEXT_CHUNK sibling chain follows *file reading order* across the whole
    # symbol list (module chunk(s), then every def in source order) -- not
    # nesting, which HAS_CHUNK already encodes via parent_symbol_id/file_id.
    for prev, nxt in zip(ordered, ordered[1:]):
        prev.next_chunk_id = nxt.id
        nxt.prev_chunk_id = prev.id

    file.imports = imports_raw
    return FileChunkResult(file=file, symbols=ordered, import_table=import_table)


# --- retrieval-quality helpers (advanced extraction payoff) -----------------
# Raw source is a poor embedding target: all-MiniLM-L6-v2 truncates at 256
# tokens and wasn't trained heavily on code, and sklearn's default TF-IDF
# tokenizer treats `get_driver` as one opaque token. Both problems are fixed
# by embedding a natural-language "card" built from the structured extraction
# above (name/signature/docstring/callers) instead of the raw body, and by
# emitting identifier-split tokens into the sparse text.

_CAMEL_RE = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z0-9]+')


def split_identifier(ident: str) -> list[str]:
    parts = re.split(r'[_\W]+', ident)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        out.extend(m.lower() for m in _CAMEL_RE.findall(p))
    return out


def build_embedding_text(symbol: CodeSymbolNode, file: CodeFileNode) -> str:
    lines = [f"{symbol.kind}: {symbol.qualified_name}"]
    if symbol.signature:
        lines.append(f"signature: {symbol.name}{symbol.signature}")
    if symbol.base_classes:
        lines.append(f"inherits from: {', '.join(symbol.base_classes)}")
    lines.append(f"in file: {file.path}")
    if symbol.decorators:
        lines.append(f"decorators: {', '.join(symbol.decorators)}")
    if symbol.docstring:
        lines.append(symbol.docstring)
    if symbol.calls_raw:
        lines.append("calls: " + ", ".join(sorted(set(symbol.calls_raw))[:20]))
    card = "\n".join(lines)
    body = symbol.text
    return f"{card}\n\ncode:\n{body}"


def build_sparse_text(symbol: CodeSymbolNode, file: CodeFileNode) -> str:
    tokens = split_identifier(symbol.name)
    for call in symbol.calls_raw:
        tokens.extend(split_identifier(call.split(".")[-1]))
    for base in symbol.base_classes:
        tokens.extend(split_identifier(base))
    return build_embedding_text(symbol, file) + "\n" + " ".join(tokens)
