"""CLI entrypoint for the full ingestion pipeline.

Usage:
  python scripts/run_ingest.py                      # all 13 topics, 1 book each
  python scripts/run_ingest.py --topics flight_rules electrical --max-books 2
  python scripts/run_ingest.py --no-connection-finder   # skip the LLM pass (fast, structure+similarity only)
"""
import argparse
import json

from graph_ops.pipeline import run_full_ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--max-books", type=int, default=1)
    parser.add_argument("--no-connection-finder", action="store_true")
    args = parser.parse_args()

    report = run_full_ingest(
        topics=args.topics,
        max_books_per_topic=args.max_books,
        run_connection_finder=not args.no_connection_finder,
    )
    print(json.dumps({
        "books": report.books,
        "topics_with_no_hits": report.topics_with_no_hits,
        "total_chunks": report.total_chunks,
        "connection_pass": report.connection_pass,
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }, indent=2))
