"""Central config, loaded from environment / .env. No hardcoded secrets."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrags123")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "graph_chunks")

# --- Embeddings ---
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "gemini"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CONNECTION_MODEL = os.getenv("OLLAMA_CONNECTION_MODEL", "gemma4:latest")
OLLAMA_ANSWER_MODEL = os.getenv("OLLAMA_ANSWER_MODEL", "gemma4:latest")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CONNECTION_MODEL = os.getenv("GEMINI_CONNECTION_MODEL", "gemini-flash-latest")
GEMINI_ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-flash-latest")

# --- Flask ---
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
FLASK_PORT = int(os.getenv("FLASK_PORT", "8000"))

# --- Ingestion ---
GUTENBERG_USER_AGENT = os.getenv(
    "GUTENBERG_USER_AGENT", "graph-rags-ingest/1.0 (educational research pipeline)"
)

# Chunking / graph-build tuning (kept here, not hardcoded deep in modules)
CHUNK_TARGET_WORDS = int(os.getenv("CHUNK_TARGET_WORDS", "180"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))
SIMILARITY_EDGE_THRESHOLD = float(os.getenv("SIMILARITY_EDGE_THRESHOLD", "0.55"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "8"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.4"))

# gemma4:latest is a large local model (~30-40s/call) -- cap how many candidate
# pairs the connection-finder judges per topic so a full-corpus run stays in
# the tens-of-minutes range rather than hours. Raise via env if you have time
# (or a faster model) to spare.
CONNECTION_FINDER_MAX_PAIRS = int(os.getenv("CONNECTION_FINDER_MAX_PAIRS", "12"))

# --- Code ingestion (ingestion/code_chunker.py, graph_ops/code_pipeline.py) ---
CODE_SUPPORTED_EXTENSIONS = os.getenv("CODE_SUPPORTED_EXTENSIONS", ".py").split(",")
CODE_EXCLUDE_DIRS = set(
    os.getenv(
        "CODE_EXCLUDE_DIRS",
        ".git,.venv,venv,__pycache__,node_modules,.mypy_cache,.pytest_cache,build,dist,.tox",
    ).split(",")
)
# Hard cap on a single symbol's stored source text (chars) -- keeps a giant
# generated function from blowing up Neo4j/Qdrant payload size. The symbol
# itself is never split (a function is the atomic semantic unit); only its
# stored text is truncated past this.
CODE_CHUNK_MAX_BODY_CHARS = int(os.getenv("CODE_CHUNK_MAX_BODY_CHARS", "4000"))
# Loose (not-in-any-def) top-level statements -- module docstring, constants,
# script bodies -- are packed into "module" symbol(s) up to this line budget,
# same spirit as CHUNK_TARGET_WORDS for prose.
CODE_MODULE_CHUNK_MAX_LINES = int(os.getenv("CODE_MODULE_CHUNK_MAX_LINES", "60"))
# Skip files bigger than this so a vendored/minified/data file can't stall ingest.
CODE_MAX_FILE_BYTES = int(os.getenv("CODE_MAX_FILE_BYTES", "512_000"))
