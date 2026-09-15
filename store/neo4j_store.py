"""Neo4j adapter. Node/edge shape is defined by graphbuild.node_schema;
this module only knows how to persist and query that shape.

`rel_type` / node `type` are only ever interpolated into Cypher after being
validated as members of RelationType / NodeType (callers must construct
those enums first, which raises ValueError on anything else) -- never raw
user strings, so this is not a Cypher-injection vector.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from neo4j import GraphDatabase

import config
from graphbuild.node_schema import GraphEdge, GraphNode, NodeType, RelationType


@lru_cache(maxsize=1)
def get_driver():
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


def close_driver() -> None:
    get_driver().close()


def init_schema() -> None:
    with get_driver().session() as session:
        for node_type in NodeType:
            session.run(
                f"CREATE CONSTRAINT {node_type.value.lower()}_id IF NOT EXISTS "
                f"FOR (n:{node_type.value}) REQUIRE n.id IS UNIQUE"
            )
        session.run("CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text, n.label]")
        session.run("CREATE FULLTEXT INDEX concept_ft IF NOT EXISTS FOR (n:Concept) ON EACH [n.label, n.description]")


def upsert_node(node: GraphNode) -> None:
    props = node.to_dict()
    props.pop("metadata", None)
    for k, v in (node.metadata or {}).items():
        props[f"meta_{k}"] = v
    label = props.pop("type")
    with get_driver().session() as session:
        session.run(
            f"MERGE (n:{label} {{id: $id}}) SET n += $props",
            id=node.id,
            props=props,
        )


def upsert_nodes(nodes: list[GraphNode]) -> None:
    by_label: dict[str, list[dict]] = {}
    for node in nodes:
        props = node.to_dict()
        props.pop("metadata", None)
        for k, v in (node.metadata or {}).items():
            props[f"meta_{k}"] = v
        label = props.pop("type")
        by_label.setdefault(label, []).append(props)
    with get_driver().session() as session:
        for label, rows in by_label.items():
            session.run(
                f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row",
                rows=rows,
            )


def upsert_edge(edge: GraphEdge) -> None:
    props = edge.to_dict()
    props.pop("metadata", None)
    for k, v in (edge.metadata or {}).items():
        props[f"meta_{k}"] = v
    rel_type = props.pop("rel_type")
    with get_driver().session() as session:
        session.run(
            "MATCH (a {id: $source_id}), (b {id: $target_id}) "
            f"MERGE (a)-[r:{rel_type} {{id: row_id}}]->(b) SET r += $props",
            source_id=edge.source_id,
            target_id=edge.target_id,
            row_id=edge.id,
            props=props,
        )


def upsert_edges(edges: list[GraphEdge]) -> None:
    by_type: dict[str, list[dict]] = {}
    for edge in edges:
        props = edge.to_dict()
        props.pop("metadata", None)
        for k, v in (edge.metadata or {}).items():
            props[f"meta_{k}"] = v
        rel_type = props.pop("rel_type")
        by_type.setdefault(rel_type, []).append(props)
    with get_driver().session() as session:
        for rel_type, rows in by_type.items():
            session.run(
                "UNWIND $rows AS row MATCH (a {id: row.source_id}), (b {id: row.target_id}) "
                f"MERGE (a)-[r:{rel_type} {{id: row.id}}]->(b) SET r += row",
                rows=rows,
            )


def get_node(node_id: str) -> Optional[dict]:
    with get_driver().session() as session:
        rec = session.run("MATCH (n {id: $id}) RETURN n, labels(n) AS labels", id=node_id).single()
        if not rec:
            return None
        node = dict(rec["n"])
        node["_labels"] = rec["labels"]
        return node


def get_node_neighborhood(node_id: str, hops: int = 1, limit: int = 50) -> dict:
    """Returns {node, relationships: [{edge, neighbor}]} for the UI's node-detail page."""
    with get_driver().session() as session:
        center = session.run("MATCH (n {id: $id}) RETURN n, labels(n) AS labels", id=node_id).single()
        if not center:
            return {"node": None, "relationships": []}
        node = dict(center["n"])
        node["_labels"] = center["labels"]

        rows = session.run(
            f"""
            MATCH (n {{id: $id}})-[r]-(m)
            RETURN r, type(r) AS rel_type, startNode(r).id = $id AS outgoing, m, labels(m) AS m_labels
            LIMIT $limit
            """,
            id=node_id,
            limit=limit,
        )
        relationships = []
        for row in rows:
            edge = dict(row["r"])
            edge["rel_type"] = row["rel_type"]
            neighbor = dict(row["m"])
            neighbor["_labels"] = row["m_labels"]
            relationships.append({
                "edge": edge,
                "outgoing": row["outgoing"],
                "neighbor": neighbor,
            })
        return {"node": node, "relationships": relationships}


def list_nodes(
    node_type: Optional[str] = None,
    topic: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    label_clause = f":{node_type}" if node_type else ""
    where_clauses = []
    params: dict[str, Any] = {"limit": limit}
    if topic:
        where_clauses.append("n.topic = $topic")
        params["topic"] = topic
    if search:
        where_clauses.append("(toLower(n.label) CONTAINS toLower($search) OR toLower(coalesce(n.text,'')) CONTAINS toLower($search))")
        params["search"] = search
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    query = f"MATCH (n{label_clause}) {where} RETURN n, labels(n) AS labels ORDER BY n.created_at DESC LIMIT $limit"
    with get_driver().session() as session:
        rows = session.run(query, **params)
        out = []
        for row in rows:
            d = dict(row["n"])
            d["_labels"] = row["labels"]
            out.append(d)
        return out


def graph_stats() -> dict:
    with get_driver().session() as session:
        node_counts = session.run(
            "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c"
        ).data()
        rel_counts = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c"
        ).data()
        return {
            "nodes": {row["l"]: row["c"] for row in node_counts},
            "relationships": {row["t"]: row["c"] for row in rel_counts},
        }


def create_relationship(
    source_id: str,
    target_id: str,
    rel_type: RelationType,
    weight: float = 1.0,
    explanation: str = "",
    created_by: str = "user",
) -> GraphEdge:
    from graphbuild.node_schema import EdgeMethod, make_edge

    method = EdgeMethod.MANUAL if created_by.startswith("user") else EdgeMethod.LLM_CONNECTION_FINDER
    edge = make_edge(source_id, target_id, rel_type, method, weight=weight, explanation=explanation, created_by=created_by)
    upsert_edge(edge)
    return edge


def create_node(node: GraphNode) -> None:
    upsert_node(node)


def delete_node(node_id: str) -> None:
    with get_driver().session() as session:
        session.run("MATCH (n {id: $id}) DETACH DELETE n", id=node_id)


def merge_nodes(keep_id: str, absorb_ids: list[str]) -> dict:
    """Rewires every relationship pointing at/from `absorb_ids` onto `keep_id`,
    unions their text/metadata into keep_id, then deletes the absorbed nodes.
    Neo4j-side only -- callers (graph_ops) must also clean up Qdrant points
    for the absorbed chunk ids so the two stores don't drift apart."""
    with get_driver().session() as session:
        for absorb_id in absorb_ids:
            if absorb_id == keep_id:
                continue
            session.run(
                """
                MATCH (keep {id: $keep_id})
                MATCH (absorbed {id: $absorb_id})-[r]->(other)
                WHERE other.id <> $keep_id
                CALL apoc.merge.relationship(keep, type(r), {}, properties(r), other, {}) YIELD rel
                RETURN count(rel)
                """,
                keep_id=keep_id, absorb_id=absorb_id,
            )
            session.run(
                """
                MATCH (keep {id: $keep_id})
                MATCH (other)-[r]->(absorbed {id: $absorb_id})
                WHERE other.id <> $keep_id
                CALL apoc.merge.relationship(other, type(r), {}, properties(r), keep, {}) YIELD rel
                RETURN count(rel)
                """,
                keep_id=keep_id, absorb_id=absorb_id,
            )
            session.run(
                """
                MATCH (keep {id: $keep_id}), (absorbed {id: $absorb_id})
                SET keep.merged_from = coalesce(keep.merged_from, []) + [absorb_id],
                    keep.aliases = coalesce(keep.aliases, []) + [absorbed.label]
                """,
                keep_id=keep_id, absorb_id=absorb_id,
            )
            session.run("MATCH (absorbed {id: $absorb_id}) DETACH DELETE absorbed", absorb_id=absorb_id)
    return {"keep_id": keep_id, "absorbed": [a for a in absorb_ids if a != keep_id]}
