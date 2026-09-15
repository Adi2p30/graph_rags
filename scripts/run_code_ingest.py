"""CLI entrypoint for code-graph ingestion.

Usage:
  python scripts/run_code_ingest.py .                                  # ingest this repo itself
  python scripts/run_code_ingest.py /path/to/other/repo --repo myrepo
  python scripts/run_code_ingest.py . --topic code --connection-finder  # also run the LLM pass
"""
import argparse
import json
from pathlib import Path

from graph_ops.code_pipeline import run_code_ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", help="Path to the source tree to ingest")
    parser.add_argument("--repo", default=None, help="Repo name stored on each CodeFile node (default: dir name)")
    parser.add_argument("--topic", default="code")
    parser.add_argument("--connection-finder", action="store_true", help="Also run the LLM connection-finder pass")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    report = run_code_ingest(
        repo_root=repo_root,
        repo=args.repo or repo_root.resolve().name,
        topic=args.topic,
        run_connection_finder=args.connection_finder,
    )
    print(json.dumps({
        "files_ingested": len(report.files),
        "files_skipped": report.files_skipped,
        "total_symbols": report.total_symbols,
        "edges_created": report.edges_created,
        "sparse_index_mode": report.sparse_index_mode,
        "elapsed_seconds": round(report.elapsed_seconds, 1),
    }, indent=2))
