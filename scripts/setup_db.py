"""Initializes Neo4j constraints/indexes and the Qdrant collection.
Run once after `docker compose up -d`."""
from store import neo4j_store, qdrant_store

if __name__ == "__main__":
    neo4j_store.init_schema()
    print("Neo4j constraints/indexes ready.")
    qdrant_store.ensure_collection()
    print("Qdrant collection ready:", qdrant_store.collection_stats())
