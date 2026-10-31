"""Download Khmer Wikipedia lead sections (resumable). Usage: python scripts/download_wikipedia.py [--limit N]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from khmer_nlp.data.wikipedia import download, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N article ids (for smoke tests)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "km_wikipedia_leads.jsonl")
    args = ap.parse_args()
    summary = download(args.out, limit=args.limit)
    meta = ROOT / "data" / "raw" / "download_summary.json"
    meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest_rows"] = write_manifest(args.out, ROOT / "data" / "manifest" / "km_wikipedia_manifest.csv")
    meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
