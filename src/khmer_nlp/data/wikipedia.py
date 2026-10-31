"""Download lead sections of Khmer Wikipedia articles with revision and category metadata.

Wikipedia text is licensed CC BY-SA 4.0 (reported by the site itself through ``meta=siteinfo&siprop=rightsinfo``).
Only the lead section (``exintro``) is fetched because the full-text extract API returns one page per request.
Raw text is written to a git-ignored file; only a metadata manifest (no article text) is meant to be committed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import requests

API = "https://km.wikipedia.org/w/api.php"
USER_AGENT = "khmer-nlp-research/0.1 (research prototype; https://github.com/THYCHANTHA)"
BATCH = 20


def _get(session: requests.Session, params: dict, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(API, params={**params, "format": "json", "formatversion": 2}, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"API request failed after {retries} attempts: {last}")


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def fetch_rights(session: requests.Session) -> dict:
    data = _get(session, {"action": "query", "meta": "siteinfo", "siprop": "rightsinfo|statistics"})
    return data["query"]


def list_article_ids(session: requests.Session) -> list[int]:
    """All non-redirect main-namespace page ids, in API (title) order."""
    ids: list[int] = []
    cont: dict = {}
    while True:
        data = _get(
            session,
            {
                "action": "query",
                "list": "allpages",
                "apnamespace": 0,
                "apfilterredir": "nonredirects",
                "aplimit": 500,
                **cont,
            },
        )
        ids.extend(p["pageid"] for p in data["query"]["allpages"])
        cont = data.get("continue", {})
        if not cont:
            return ids


def fetch_batch(session: requests.Session, page_ids: list[int]) -> list[dict]:
    """Fetch one batch, following API continuation until every property is complete."""
    merged: dict[int, dict] = {}
    cont: dict = {}
    while True:
        data = _get(
            session,
            {
                "action": "query",
                "pageids": "|".join(map(str, page_ids)),
                "prop": "extracts|categories|revisions|info",
                "exintro": 1,
                "explaintext": 1,
                "exlimit": BATCH,
                "cllimit": "max",
                "clshow": "!hidden",
                "rvprop": "ids|timestamp",
                "inprop": "url",
                **cont,
            },
        )
        for page in data["query"]["pages"]:
            cur = merged.setdefault(page["pageid"], {"pageid": page["pageid"], "categories": []})
            for key, value in page.items():
                if key == "categories":
                    cur["categories"].extend(c["title"] for c in value)
                else:
                    cur[key] = value
        cont = data.get("continue", {})
        if not cont:
            break
    out = []
    for p in merged.values():
        rev = (p.get("revisions") or [{}])[0]
        out.append(
            {
                "pageid": p["pageid"],
                "title": p.get("title", ""),
                "url": p.get("fullurl", ""),
                "history_url": p.get("fullurl", "").replace("/wiki/", "/w/index.php?title=", 1) + "&action=history"
                if "/wiki/" in p.get("fullurl", "")
                else "",
                "revid": rev.get("revid"),
                "timestamp": rev.get("timestamp"),
                "categories": sorted(set(p["categories"])),
                "text": p.get("extract", ""),
            }
        )
    return out


def chunks(items: list[int], size: int) -> Iterator[list[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def download(out_path: Path, limit: int | None = None, pause: float = 0.2) -> dict:
    """Resumable download to a JSONL file. Returns a small run summary (including licence info)."""
    session = new_session()
    site = fetch_rights(session)
    ids = list_article_ids(session)
    if limit:
        ids = ids[:limit]
    done: set[int] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            done = {json.loads(line)["pageid"] for line in fh if line.strip()}
    todo = [i for i in ids if i not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for batch in chunks(todo, BATCH):
            for rec in fetch_batch(session, batch):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(pause)
    return {
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "api": API,
        "rights": site["rightsinfo"],
        "site_statistics": site["statistics"],
        "article_ids_listed": len(ids),
        "already_present": len(done),
        "fetched_now": len(todo),
    }


def write_manifest(raw_path: Path, manifest_path: Path) -> int:
    """Metadata-only manifest (no article text): enough to retrieve the exact revisions again and to attribute them."""
    import csv

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with raw_path.open(encoding="utf-8") as src, manifest_path.open("w", encoding="utf-8", newline="") as dst:
        w = csv.writer(dst)
        w.writerow(["pageid", "title", "url", "history_url", "revid", "timestamp", "lead_chars"])
        for line in src:
            if not line.strip():
                continue
            r = json.loads(line)
            w.writerow(
                [r["pageid"], r["title"], r["url"], r["history_url"], r["revid"], r["timestamp"], len(r["text"])]
            )
            n += 1
    return n
