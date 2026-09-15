"""Fetches Project Gutenberg books for each domain topic.

Earlier versions of this module did a live keyword search per topic
(`www.gutenberg.org/ebooks/search`). That produced real off-topic hits --
"The Happy Prince, and Other Tales" for trajectory_guidance_propulsion,
"Wuthering Heights" for thermal, "Principles of Orchestration" for
ground_instrumentation -- because Gutenberg's search matches loosely against
title/author/subject text, and a broad fallback query (needed since Gutenberg's
pre-1929 catalog has no direct match for modern terms like "guidance navigation
control") will happily return fiction.

Per explicit project requirement, every ingested book must actually be about
flight/aviation/aerospace. So this module now draws from a fixed, individually
verified list (`TOPIC_BOOK_IDS`) instead of searching: each id below was
downloaded and checked -- title, author, and the opening text of the book body
-- before being added. No live search, no risk of a title-keyword coincidence
pulling in an unrelated novel.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import config
from ingestion.http_fetch import get

TEXT_URL_CANDIDATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

# Verified aviation/aerospace/flight-relevant books (title + author + a real
# excerpt of the body text were checked for each -- see project chat history).
# gid -> (title, author)
VERIFIED_BOOKS: dict[int, tuple[str, str]] = {
    874: ("A History of Aeronautics", "Evelyn Charles Vivian"),
    34815: ("Jane's All the World's Aircraft, 1913", "Fred T. Jane"),
    60277: ("Aerial Navigation", "Albert Francis Zahm"),
    47981: ("Langley Memoir on Mechanical Flight, Parts I and II", "S. P. Langley"),
    27557: ("Learning to Fly: A Practical Manual for Beginners", "Claude Grahame-White"),
    25420: ("The Early History of the Airplane", "Orville Wright"),
    907: ("Flying Machines: Construction and Operation", "William J. Jackman"),
    793: ("Aeroplanes and Dirigibles of War", "Frederick Arthur Ambrose Talbot"),
    66702: ("Wireless Telegraphy and Telephony Simply Explained", "Alfred Powell Morgan"),
    40268: ("Significant Achievements in Space Bioscience 1958-1964", "NASA"),
    57973: ("The Hurricane Hunters", "Ivan Ray Tannehill"),
    32570: ("Zeppelin: The Story of a Great Achievement", "Harry Vissering"),
    73142: ('"We"', "Charles A. Lindbergh"),
    38187: ("Aviation Engines: Design—Construction—Operation and Repair", "Victor Wilfred Page"),
}

# Each topic draws only from this verified pool, ordered by relevance.
TOPIC_BOOK_IDS: dict[str, list[int]] = {
    "flight_rules": [27557, 73142],
    "flight_operations": [38187, 57973],
    "ground_instrumentation": [34815, 60277],
    "trajectory_guidance_propulsion": [47981, 38187],
    "data_systems": [66702],
    "guidance_navigation_control": [25420, 60277],
    "electrical": [66702, 907],
    "mechanical": [793, 907],
    "communications": [66702],
    "aeronautical": [874, 34815, 32570],
    "space_environment": [40268],
    "post_landing_life_support": [40268],
    "thermal": [38187],
}


@dataclass
class GutenbergHit:
    gutenberg_id: int
    title: str


def download_text(gutenberg_id: int) -> str | None:
    for template in TEXT_URL_CANDIDATES:
        url = template.format(id=gutenberg_id)
        status, body = get(url, retries=2)
        if status == 200 and body:
            return body.decode("utf-8", errors="ignore")
    return None


def fetch_topic(topic: str, max_books: int = 1) -> list[dict]:
    """Downloads up to `max_books` verified aviation/aerospace books for a topic.
    Returns list of dicts: {gutenberg_id, title, topic, query_used, raw_path}."""
    ids = TOPIC_BOOK_IDS.get(topic, [])[:max_books]
    results = []
    topic_dir = config.RAW_DIR / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    for gid in ids:
        title, author = VERIFIED_BOOKS[gid]
        text = download_text(gid)
        if not text or len(text) < 2000:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
        raw_path = topic_dir / f"{gid}_{slug}.txt"
        raw_path.write_text(text, encoding="utf-8")
        results.append({
            "gutenberg_id": gid,
            "title": title,
            "topic": topic,
            "query_used": f"curated:{author}",
            "raw_path": str(raw_path),
        })
        time.sleep(0.5)
    return results


def fetch_all_topics(max_books_per_topic: int = 1) -> dict[str, list[dict]]:
    coverage: dict[str, list[dict]] = {}
    for topic in TOPIC_BOOK_IDS:
        coverage[topic] = fetch_topic(topic, max_books=max_books_per_topic)
    return coverage
