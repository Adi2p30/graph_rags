"""LLM connection-finder: takes statistically-similar chunk pairs (from the
sparse+dense hybrid pass) and judges whether a real, typed relationship
exists -- and whether the pair actually instantiates a shared Concept node
that should tie multiple chunks/books together, rather than just a one-off
edge. This is the "use an LLM to find connections" + "make new concepts"
requirement; the pure-statistics pass only ever produces untyped SIMILAR_TO
candidates, never the final typed graph.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from graphbuild.node_schema import (
    ChunkNode, ConceptNode, EdgeMethod, GraphEdge, RelationType, new_id, make_edge,
)
from llm.base import LLMProvider, get_provider

ALLOWED_REL_TYPES = {
    "RELATES_TO", "SUPPORTS", "CONTRADICTS", "ELABORATES", "PREREQUISITE_OF",
}

SYSTEM_PROMPT = """You are a domain analyst building a knowledge graph out of technical/engineering \
book excerpts about flight systems (flight rules, operations, guidance/navigation/control, \
propulsion, electrical, mechanical, communications, thermal, life support, etc).

Given two text passages that scored as statistically similar, decide:
1. Whether they are actually semantically related (not just sharing common words).
2. If related, the single best relationship type from: RELATES_TO, SUPPORTS, CONTRADICTS, \
ELABORATES, PREREQUISITE_OF.
3. A confidence score 0.0-1.0.
4. A one-sentence explanation grounded in the actual text (quote or paraphrase specifics -- \
never a generic statement like "both discuss the topic").
5. Optionally, if both passages are really instances of one recurring underlying idea/concept \
worth its own node (e.g. "gyroscopic precession", "static stability"), name it concisely \
(3-6 words) in concept_name; otherwise use null.

Reply with ONLY this JSON shape:
{"related": true|false, "rel_type": "RELATES_TO"|null, "confidence": 0.0-1.0, \
"explanation": "...", "concept_name": "..."|null, "concept_description": "..."|null}
"""


@dataclass
class ConnectionResult:
    edge: GraphEdge | None
    concept: ConceptNode | None
    concept_edges: list[GraphEdge]


def _build_prompt(a: ChunkNode, b: ChunkNode) -> str:
    return f"""PASSAGE A (topic: {a.topic}, book: {a.metadata.get('book_title', '?')}):
{a.text[:900]}

PASSAGE B (topic: {b.topic}, book: {b.metadata.get('book_title', '?')}):
{b.text[:900]}

Judge the relationship per the system instructions."""


def judge_pair(
    a: ChunkNode,
    b: ChunkNode,
    provider: LLMProvider | None = None,
) -> dict | None:
    provider = provider or get_provider("connection")
    result = provider.complete_json(_build_prompt(a, b), system=SYSTEM_PROMPT)
    if result is None:
        return None
    if result.get("rel_type") not in ALLOWED_REL_TYPES:
        result["related"] = False
    return result


def process_candidates(
    candidates: list[tuple[ChunkNode, ChunkNode, float]],
    existing_concepts: dict[str, ConceptNode] | None = None,
    provider: LLMProvider | None = None,
) -> tuple[list[GraphEdge], list[ConceptNode], list[GraphEdge]]:
    """Returns (typed_edges, new_concepts, concept_edges).
    `existing_concepts` maps lowercased concept name -> ConceptNode, shared across
    calls so repeated mentions attach to the same concept instead of duplicating it."""
    provider = provider or get_provider("connection")
    concepts = existing_concepts if existing_concepts is not None else {}
    typed_edges: list[GraphEdge] = []
    new_concepts: list[ConceptNode] = []
    concept_edges: list[GraphEdge] = []

    for chunk_a, chunk_b, score in candidates:
        judgment = judge_pair(chunk_a, chunk_b, provider=provider)
        if not judgment or not judgment.get("related"):
            continue

        rel_type = RelationType(judgment["rel_type"])
        edge = make_edge(
            chunk_a.id, chunk_b.id, rel_type, EdgeMethod.LLM_CONNECTION_FINDER,
            weight=float(judgment.get("confidence", 0.5)),
            explanation=judgment.get("explanation", ""),
            created_by=f"llm:{provider.model_name}",
            similarity_score=score,
        )
        typed_edges.append(edge)

        concept_name = judgment.get("concept_name")
        if concept_name:
            key = concept_name.strip().lower()
            concept = concepts.get(key)
            if concept is None:
                concept = ConceptNode(
                    id=new_id("concept"),
                    type=None,
                    label=concept_name.strip(),
                    topic=chunk_a.topic,
                    description=judgment.get("concept_description", ""),
                    created_by=f"llm:{provider.model_name}",
                )
                concepts[key] = concept
                new_concepts.append(concept)
            for chunk in (chunk_a, chunk_b):
                if chunk.id not in concept.source_chunk_ids:
                    concept.source_chunk_ids.append(chunk.id)
                    concept_edges.append(make_edge(
                        chunk.id, concept.id, RelationType.PART_OF_CONCEPT, EdgeMethod.LLM_CONNECTION_FINDER,
                        weight=float(judgment.get("confidence", 0.5)),
                        explanation=f"Instance of concept '{concept.label}'",
                        created_by=f"llm:{provider.model_name}",
                    ))

    return typed_edges, new_concepts, concept_edges
