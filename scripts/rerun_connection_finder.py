"""Re-runs the AI connection-finder pass across all 13 topics (used once here
to regenerate LLM-derived edges/concepts with gemma after purging qwen-tainted
data; harmless to run again any time -- it's the same function the GUI's
AI Tools page triggers).

Usage: python -m scripts.rerun_connection_finder [topic ...]
With no args, runs all 13 topics.
"""
import json
import sys

from graph_ops.edge_ops import run_connection_finder_pass
from ingestion.gutenberg_fetch import TOPIC_BOOK_IDS

if __name__ == "__main__":
    topics = sys.argv[1:] or list(TOPIC_BOOK_IDS.keys())
    results = {}
    for topic in topics:
        results[topic] = run_connection_finder_pass(topic=topic)
        print(topic, results[topic], flush=True)
    print(json.dumps(results, indent=2))
