# Flight Systems GraphRAG

Builds a knowledge graph out of public-domain (Project Gutenberg) books across 13 flight-systems
domains -- flight rules, flight operations, ground instrumentation, trajectory/guidance/propulsion,
data systems, GNC, electrical, mechanical, communications, aeronautical, space environment,
post-landing life support, thermal -- then lets you query it with GraphRAG and edit it through a
Flask GUI.

Two stores, not one: **Neo4j** holds the typed graph structure (Book/Section/Chunk/Concept nodes,
typed relationships with confidence + explanation metadata). **Qdrant** holds the same chunks/concepts
as hybrid dense+sparse vectors for retrieval. Neither is "the" database on its own -- `graph_ops/`
is the only code that's allowed to write to both, and it always does so together (see `node_ops.merge_nodes`).

## Architecture

```
ingestion/        Gutenberg search + download, boilerplate stripping, structure-aware chunking
graphbuild/        node_schema.py (the shared Node/Edge dataclasses), embeddings (sentence-transformers),
                    sparse (TF-IDF), similarity (hybrid cosine -> candidate edges)
llm/                provider abstraction (Ollama local / Gemini cloud), connection_finder (LLM #1:
                    judges candidate pairs, proposes Concept nodes), graphrag_answerer (LLM #2:
                    hybrid retrieval + graph traversal + grounded answer)
store/              neo4j_store.py, qdrant_store.py -- the only modules that touch the DB drivers
graph_ops/          node_ops (create/merge, dual-store), edge_ops (manual + AI relationship creation),
                    pipeline.py (full ingest orchestration)
app/                Flask GUI (dashboard, node browser/detail, visual graph explorer, GraphRAG query
                    page, AI tools, ingest trigger) -- this *is* the frontend, no separate SPA
```

### Node/edge schema (`graphbuild/node_schema.py`)

Four node types: `Book`, `Section`, `Chunk`, `Concept`. Every node carries `topic`, `created_by`,
`created_at`, and a free-form `metadata` dict. Concepts are synthesized by the LLM connection-finder
(or manually via the GUI) to tie together chunks that instantiate the same recurring idea.

Edges carry `rel_type`, `method` (`structural` / `hybrid_similarity` / `llm_connection_finder` /
`manual`), a `weight` (similarity score or LLM confidence), and an `explanation` string the GUI
shows next to each relationship -- an edge with no textual justification means something is wrong.

### How the graph gets built

1. **Structural pass** (no ML): chapter/section headings detected from plain-text layout ->
   `Book -HAS_SECTION-> Section -HAS_CHUNK-> Chunk`, plus `Chunk -NEXT_CHUNK-> Chunk` reading order.
2. **Statistical pass**: sentence-transformers dense embeddings + corpus-wide TF-IDF sparse vectors,
   combined into a weighted hybrid cosine score (`config.DENSE_WEIGHT` / `SPARSE_WEIGHT`). Adjacent
   chunks are excluded (that's already `NEXT_CHUNK`) so this only surfaces non-trivial similarity.
   This pass never writes a typed edge by itself -- it only produces candidates.
3. **LLM pass** (`llm/connection_finder.py`, Ollama `gemma3:1b` by default): judges each candidate
   pair, picks a relationship type (`RELATES_TO`/`SUPPORTS`/`CONTRADICTS`/`ELABORATES`/
   `PREREQUISITE_OF`), and optionally names a shared `Concept` both chunks instantiate.
4. **GraphRAG answering** (`llm/graphrag_answerer.py`, a *different* Ollama model by default,
   `gemma3:4b`): hybrid-searches Qdrant for seed nodes, traverses 1-2 hops out in Neo4j, and asks
   the model to answer grounded in that assembled context, citing node ids.

### AI graph-editing operations (also triggerable from the GUI)

- **Create node** -- manual Concept or note-chunk (`/nodes/new`).
- **Merge nodes** -- `graph_ops/node_ops.merge_nodes`: rewires every Neo4j relationship onto the
  kept node, deletes the absorbed Qdrant points, and re-embeds the kept node's combined text so
  retrieval still surfaces it (`/nodes/merge`).
- **Create relationship** -- manual (`/relationships/new`) or AI-driven
  (`/ai`, re-runs the statistical + LLM pass over whatever's currently in the graph, not just at
  ingest time) -- both can mint new Concept nodes.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # 3.12, not 3.14 -- torch wheels lag new CPython minors
pip install -r requirements.txt
cp .env.example .env

docker compose up -d          # Neo4j (7474/7687) + Qdrant (6333/6334)
python -m scripts.setup_db    # constraints/indexes + Qdrant collection

# pull the two Ollama models this project uses by default, if you don't have them:
ollama pull gemma3:1b
ollama pull gemma3:4b

python -m scripts.run_ingest                 # all 13 topics, 1 book each
# or, to skip the LLM pass while iterating on ingestion:
python -m scripts.run_ingest --no-connection-finder

python run.py                 # Flask GUI at http://localhost:8000
```

`LLM_PROVIDER` in `.env` switches both LLM roles between `ollama` (default, local, free) and
`gemini` (needs `GEMINI_API_KEY`) -- see `.env.example` for the per-role model names.

Only gemma (Ollama) or gemini (cloud) models are used in this project -- qwen is deliberately
excluded even though it's installed locally, per explicit project constraint.

## Known limitations (by design, not oversight)

- **Gutenberg coverage is uneven.** The corpus is almost entirely pre-1929. Terms like "guidance
  navigation control" or "post-landing life support" have no direct hits; `ingestion/gutenberg_fetch.py`
  falls back through period-appropriate broader queries per topic (e.g. "gyroscope", "physiology")
  and the ingest report lists which topics came back empty (`topics_with_no_hits`) rather than
  silently backfilling with off-topic books.
- **TF-IDF is refit corpus-wide on every `run_full_ingest` call**, not incrementally. Running ingest
  again re-embeds every chunk's sparse vector so IDF stays consistent across the whole corpus. This
  means re-running ingest is a full rebuild, not an append.
- **Single-shot batch pipeline**, not a live crawler -- there's no scheduler/queue; `/ingest` in the
  GUI runs the same one-shot function in a background thread.
