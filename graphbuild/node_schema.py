"""
Custom node/edge schema for the graph.

This is the single source of truth both storage backends serialize through:
Neo4j gets the node as labeled properties, Qdrant gets it as point payload
(alongside its dense+sparse vectors). Neither store owns the schema.
"""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


class NodeType(str, Enum):
    BOOK = "Book"
    SECTION = "Section"
    CHUNK = "Chunk"
    CONCEPT = "Concept"
    CODE_FILE = "CodeFile"


class RelationType(str, Enum):
    # structural (derived straight from document/file layout)
    HAS_SECTION = "HAS_SECTION"          # Book -> Section, Section -> Section (nesting)
    HAS_CHUNK = "HAS_CHUNK"              # Section -> Chunk; also CodeFile -> symbol, symbol -> nested symbol
    NEXT_CHUNK = "NEXT_CHUNK"            # Chunk -> Chunk (reading order / definition order)
    # similarity-derived (sparse/dense hybrid score, no LLM involved)
    SIMILAR_TO = "SIMILAR_TO"            # Chunk -> Chunk, cross-book candidate edges
    # static-analysis-derived (deterministic identifier/import resolution, no LLM)
    CALLS = "CALLS"                      # function/method symbol -> function/method symbol
    IMPORTS = "IMPORTS"                  # CodeFile -> CodeFile
    INHERITS_FROM = "INHERITS_FROM"      # class symbol -> class symbol
    # LLM-derived (connection_finder judged these)
    RELATES_TO = "RELATES_TO"            # generic typed semantic relation between chunks/concepts
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    ELABORATES = "ELABORATES"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    PART_OF_CONCEPT = "PART_OF_CONCEPT"  # Chunk -> Concept (chunk instantiates/mentions concept)
    RELATED_CONCEPT = "RELATED_CONCEPT"  # Concept -> Concept


class EdgeMethod(str, Enum):
    STRUCTURAL = "structural"
    STATIC_ANALYSIS = "static_analysis"  # deterministic code analysis (calls/imports/inheritance resolution)
    HYBRID_SIMILARITY = "hybrid_similarity"
    LLM_CONNECTION_FINDER = "llm_connection_finder"
    MANUAL = "manual"


@dataclass
class GraphNode:
    """Base fields every node carries, regardless of type."""
    id: str
    type: NodeType
    label: str                       # short human-readable title
    topic: str = ""                  # which of the 13 domain topics this belongs to
    created_by: str = "pipeline"     # "pipeline" | "llm:<model>" | "user:<email>"
    created_at: float = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class BookNode(GraphNode):
    author: str = ""
    gutenberg_id: Optional[int] = None
    source_url: str = ""
    license: str = "Public Domain (Project Gutenberg)"

    def __post_init__(self):
        self.type = NodeType.BOOK


@dataclass
class SectionNode(GraphNode):
    book_id: str = ""
    parent_section_id: Optional[str] = None
    level: int = 1                   # 1 = chapter, 2 = subsection, ...
    order_index: int = 0

    def __post_init__(self):
        self.type = NodeType.SECTION


@dataclass
class ChunkNode(GraphNode):
    book_id: str = ""
    section_id: str = ""
    text: str = ""
    word_count: int = 0
    order_index: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    def __post_init__(self):
        self.type = NodeType.CHUNK
        if self.text and not self.word_count:
            self.word_count = len(self.text.split())


@dataclass
class ConceptNode(GraphNode):
    """Not tied to one book/section — synthesized (usually by the connection-finder LLM)
    to represent an idea that recurs across multiple source chunks."""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.type = NodeType.CONCEPT


@dataclass
class CodeFileNode(GraphNode):
    """One per source file -- the code-graph analog of BookNode. Never embedded
    into Qdrant itself (nothing to retrieve at file granularity); it's the
    structural anchor that CodeSymbolNode chunks hang off of via HAS_CHUNK."""
    repo: str = ""
    path: str = ""                   # repo-relative path, e.g. "graph_ops/pipeline.py"
    module_path: str = ""            # dotted import path, e.g. "graph_ops.pipeline"
    language: str = ""
    loc: int = 0
    imports: list[str] = field(default_factory=list)   # raw "import x" / "from x import y" lines

    def __post_init__(self):
        self.type = NodeType.CODE_FILE


@dataclass
class CodeSymbolNode(GraphNode):
    """A semantic code unit (module-level summary, class, function, or method).
    Stamped with NodeType.CHUNK (not a new type) so it rides every existing
    Chunk code path for free: the `chunk_text_ft` fulltext index, the
    connection-finder pass's `node_type="Chunk"` pull, and Qdrant's
    `type == "Chunk"` retrieval filter. All fields are Neo4j-safe scalars or
    list[str] -- nothing nested (see graphbuild.node_schema module docstring
    discipline); richer structure (per-param types, raw call sites) belongs in
    the Qdrant payload, not here.
    """
    file_id: str = ""
    parent_symbol_id: Optional[str] = None   # enclosing class symbol, for methods
    kind: str = "function"                   # "module" | "class" | "function" | "method"
    language: str = ""
    name: str = ""
    qualified_name: str = ""                 # dotted path, e.g. "graph_ops.pipeline.run_full_ingest"
    signature: str = ""                      # rendered "(params) -> return_type", empty for module/class
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)   # raw superclass names, kind == "class" only
    calls_raw: list[str] = field(default_factory=list)      # raw callee expressions found in the body
    start_line: int = 0
    end_line: int = 0
    complexity: int = 0                      # branch-count proxy, see code_chunker._branch_count
    text: str = ""                           # source slice (may be truncated, see truncated flag)
    truncated: bool = False
    word_count: int = 0
    order_index: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    def __post_init__(self):
        self.type = NodeType.CHUNK
        if self.text and not self.word_count:
            self.word_count = len(self.text.split())


@dataclass
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    rel_type: RelationType
    method: EdgeMethod
    weight: float = 1.0                       # similarity score or LLM confidence, 0..1
    explanation: str = ""                      # LLM rationale / human note, shown in the UI
    created_by: str = "pipeline"
    created_at: float = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rel_type"] = self.rel_type.value
        d["method"] = self.method.value
        return d


def make_edge(
    source_id: str,
    target_id: str,
    rel_type: RelationType,
    method: EdgeMethod,
    weight: float = 1.0,
    explanation: str = "",
    created_by: str = "pipeline",
    **metadata: Any,
) -> GraphEdge:
    return GraphEdge(
        id=new_id("edge"),
        source_id=source_id,
        target_id=target_id,
        rel_type=rel_type,
        method=method,
        weight=weight,
        explanation=explanation,
        created_by=created_by,
        metadata=metadata,
    )
