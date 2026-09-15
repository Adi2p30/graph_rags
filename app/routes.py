from __future__ import annotations

import threading
import traceback

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from graph_ops import edge_ops, node_ops
from graph_ops.pipeline import run_full_ingest
from graphbuild.node_schema import NodeType, RelationType
from ingestion.gutenberg_fetch import TOPIC_BOOK_IDS
from llm.graphrag_answerer import answer as graphrag_answer
from store import neo4j_store, qdrant_store

bp = Blueprint("main", __name__)

TOPICS = list(TOPIC_BOOK_IDS.keys())

_INGEST_STATE = {"running": False, "report": None, "error": None}
_AI_STATE = {"running": False, "result": None, "error": None}


# ---------------------------------------------------------------- dashboard
@bp.route("/")
def dashboard():
    try:
        stats = neo4j_store.graph_stats()
    except Exception as e:
        stats = {"nodes": {}, "relationships": {}}
    try:
        qstats = qdrant_store.collection_stats()
    except Exception:
        qstats = {"points_count": "?", "status": "unreachable"}
    return render_template(
        "dashboard.html", stats=stats, qstats=qstats, topics=TOPICS,
        ingest_state=_INGEST_STATE,
    )


# -------------------------------------------------------------------- nodes
@bp.route("/nodes")
def browse_nodes():
    node_type = request.args.get("type") or None
    topic = request.args.get("topic") or None
    search = request.args.get("q") or None
    nodes = neo4j_store.list_nodes(node_type=node_type, topic=topic, search=search, limit=200)
    return render_template(
        "nodes.html", nodes=nodes, node_types=[t.value for t in NodeType], topics=TOPICS,
        selected_type=node_type, selected_topic=topic, search=search or "",
    )


@bp.route("/nodes/<node_id>")
def node_detail(node_id: str):
    data = neo4j_store.get_node_neighborhood(node_id, hops=1, limit=100)
    if not data["node"]:
        return render_template("not_found.html", kind="node", node_id=node_id), 404
    rel_types = [t.value for t in RelationType]
    other_nodes = neo4j_store.list_nodes(limit=300)
    return render_template(
        "node_detail.html", node=data["node"], relationships=data["relationships"],
        rel_types=rel_types, other_nodes=[n for n in other_nodes if n["id"] != node_id],
    )


@bp.route("/nodes/<node_id>/delete", methods=["POST"])
def delete_node(node_id: str):
    neo4j_store.delete_node(node_id)
    qdrant_store.delete_points([node_id])
    return redirect(url_for("main.browse_nodes"))


@bp.route("/nodes/new", methods=["GET", "POST"])
def new_node():
    if request.method == "POST":
        kind = request.form.get("kind")
        label = request.form.get("label", "").strip()
        topic = request.form.get("topic", "")
        body = request.form.get("body", "").strip()
        if kind == "concept":
            node = node_ops.create_concept_node(label, body, topic, created_by="user:manual")
        else:
            node = node_ops.create_manual_relationship_node_text(label, body, topic, created_by="user:manual")
        return redirect(url_for("main.node_detail", node_id=node.id))
    return render_template("node_new.html", topics=TOPICS)


@bp.route("/nodes/merge", methods=["GET", "POST"])
def merge_nodes_view():
    if request.method == "POST":
        keep_id = request.form.get("keep_id")
        absorb_ids = request.form.getlist("absorb_ids")
        result = node_ops.merge_nodes(keep_id, absorb_ids)
        return redirect(url_for("main.node_detail", node_id=result["keep_id"]))
    nodes = neo4j_store.list_nodes(limit=500)
    return render_template("node_merge.html", nodes=nodes)


# --------------------------------------------------------------- relationships
@bp.route("/relationships/new", methods=["GET", "POST"])
def new_relationship():
    if request.method == "POST":
        source_id = request.form.get("source_id")
        target_id = request.form.get("target_id")
        rel_type = request.form.get("rel_type")
        explanation = request.form.get("explanation", "")
        edge = edge_ops.create_relationship(source_id, target_id, rel_type, explanation, created_by="user:manual")
        if edge is None:
            return render_template("error.html", message=f"Invalid relationship type: {rel_type}"), 400
        return redirect(url_for("main.node_detail", node_id=source_id))
    nodes = neo4j_store.list_nodes(limit=500)
    rel_types = [t.value for t in RelationType]
    source_id = request.args.get("source_id", "")
    return render_template("relationship_new.html", nodes=nodes, rel_types=rel_types, source_id=source_id)


# ----------------------------------------------------------------- AI tools
@bp.route("/ai")
def ai_tools():
    return render_template("ai_tools.html", topics=TOPICS, ai_state=_AI_STATE)


@bp.route("/ai/run-connection-finder", methods=["POST"])
def run_connection_finder():
    topic = request.form.get("topic") or None

    def _run():
        _AI_STATE.update(running=True, result=None, error=None)
        try:
            result = edge_ops.run_connection_finder_pass(topic=topic)
            _AI_STATE.update(running=False, result=result, error=None)
        except Exception as e:
            _AI_STATE.update(running=False, result=None, error=f"{e}\n{traceback.format_exc()}")

    threading.Thread(target=_run, daemon=True).start()
    return redirect(url_for("main.ai_tools"))


@bp.route("/ai/status")
def ai_status():
    return jsonify(_AI_STATE)


# ------------------------------------------------------------------- ingest
@bp.route("/ingest", methods=["GET", "POST"])
def ingest():
    if request.method == "POST":
        topics = request.form.getlist("topics") or None
        max_books = int(request.form.get("max_books", 1))
        run_cf = request.form.get("run_connection_finder") == "on"

        def _run():
            _INGEST_STATE.update(running=True, report=None, error=None)
            try:
                report = run_full_ingest(topics=topics, max_books_per_topic=max_books, run_connection_finder=run_cf)
                _INGEST_STATE.update(running=False, report={
                    "books": report.books,
                    "topics_with_no_hits": report.topics_with_no_hits,
                    "total_chunks": report.total_chunks,
                    "connection_pass": report.connection_pass,
                    "elapsed_seconds": round(report.elapsed_seconds, 1),
                }, error=None)
            except Exception as e:
                _INGEST_STATE.update(running=False, report=None, error=f"{e}\n{traceback.format_exc()}")

        threading.Thread(target=_run, daemon=True).start()
        return redirect(url_for("main.dashboard"))
    return render_template("ingest.html", topics=TOPICS)


@bp.route("/ingest/status")
def ingest_status():
    return jsonify(_INGEST_STATE)


# --------------------------------------------------------------- GraphRAG
@bp.route("/query", methods=["GET", "POST"])
def query():
    result = None
    query_text = ""
    topic = request.values.get("topic") or None
    if request.method == "POST":
        query_text = request.form.get("query", "").strip()
        if query_text:
            try:
                result = graphrag_answer(query_text, topic=topic)
            except Exception as e:
                result = None
                return render_template("error.html", message=f"GraphRAG query failed: {e}"), 500
    return render_template("query.html", result=result, query_text=query_text, topics=TOPICS, topic=topic)


# --------------------------------------------------------------- graph view
@bp.route("/graph")
def graph_view():
    return render_template("graph_view.html", topics=TOPICS)


@bp.route("/api/subgraph")
def api_subgraph():
    topic = request.args.get("topic") or None
    node_type = request.args.get("type") or None
    limit = int(request.args.get("limit", 80))
    nodes = neo4j_store.list_nodes(node_type=node_type, topic=topic, limit=limit)
    node_ids = {n["id"] for n in nodes}
    edges = []
    seen_edge_ids = set()
    for n in nodes:
        nb = neo4j_store.get_node_neighborhood(n["id"], hops=1, limit=30)
        for rel in nb["relationships"]:
            neighbor_id = rel["neighbor"]["id"]
            if neighbor_id not in node_ids:
                continue
            edge = rel["edge"]
            if edge["id"] in seen_edge_ids:
                continue
            seen_edge_ids.add(edge["id"])
            src = n["id"] if rel["outgoing"] else neighbor_id
            dst = neighbor_id if rel["outgoing"] else n["id"]
            edges.append({"id": edge["id"], "from": src, "to": dst, "label": edge.get("rel_type", "")})

    vis_nodes = [
        {
            "id": n["id"],
            "label": (n.get("label") or "")[:40],
            "group": n.get("_labels", ["Node"])[0],
            "title": n.get("topic", ""),
        }
        for n in nodes
    ]
    return jsonify({"nodes": vis_nodes, "edges": edges})
