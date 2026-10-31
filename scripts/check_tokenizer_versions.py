"""Compare khmer-nltk releases on the same text (version-compatibility check).

Each version is installed with ``pip install --target <temp dir>`` (no change to the active environment) and run in a
separate process on the first N lead sections of the downloaded Wikipedia sample. Output: reports/tokenizer_versions.md.

    python scripts/check_tokenizer_versions.py --versions 1.6 1.5 1.3 1.0 --docs 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "km_wikipedia_leads.jsonl"

RUNNER = r"""
import json, sys, logging
logging.disable(logging.CRITICAL)
from khmernltk import word_tokenize
texts = json.load(open(sys.argv[1], encoding="utf-8"))
out = [[t for t in word_tokenize(x) if t.strip()] for x in texts]
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
"""


def boundaries(tokens: list[str]) -> set[int]:
    pos, out = 0, set()
    for t in tokens:
        pos += len(t)
        out.add(pos)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["1.6", "1.5", "1.3", "1.0"])
    ap.add_argument("--docs", type=int, default=200)
    args = ap.parse_args()

    from khmer_nlp.normalization.text import normalize

    texts = []
    with RAW.open(encoding="utf-8") as fh:
        for line in fh:
            t = normalize(json.loads(line).get("text", ""))[0]
            if len(t) >= 200:
                texts.append(t)
            if len(texts) >= args.docs:
                break

    results: dict[str, list[list[str]] | str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        inp = tmp_p / "texts.json"
        inp.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        runner = tmp_p / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        for v in args.versions:
            target = tmp_p / f"v{v}"
            pip = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-deps",
                    "--timeout",
                    "120",
                    "--retries",
                    "5",
                    "--target",
                    str(target),
                    f"khmer-nltk=={v}",
                ],
                capture_output=True,
                text=True,
            )
            if pip.returncode:
                results[v] = f"install failed: {pip.stderr.strip()[-200:]}"
                continue
            out = tmp_p / f"out_{v}.json"
            env = {"PYTHONPATH": str(target)}
            run = subprocess.run(
                [sys.executable, str(runner), str(inp), str(out)],
                capture_output=True,
                text=True,
                env={**__import__("os").environ, **env},
            )
            results[v] = (
                json.loads(out.read_text(encoding="utf-8"))
                if run.returncode == 0
                else f"run failed: {run.stderr.strip()[-300:]}"
            )

    ref_v = args.versions[0]
    ref = results[ref_v]
    lines = [
        "# khmer-nltk version comparison\n",
        f"{len(texts)} lead sections (normalised), reference = {ref_v}. "
        "Boundary-F1 is averaged over documents; exact = share of documents segmented identically.\n",
        "| version | status | tokens | mean boundary F1 vs ref | exact-match documents |",
        "|---|---|---|---|---|",
    ]
    for v in args.versions:
        r = results[v]
        if isinstance(r, str):
            lines.append(f"| {v} | {r} | – | – | – |")
            continue
        f1s, exact = [], 0
        for a, b in zip(ref, r, strict=True):
            ba, bb = boundaries(a), boundaries(b)
            inter = len(ba & bb)
            p, rc = inter / len(bb), inter / len(ba)
            f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
            exact += a == b
        lines.append(f"| {v} | ok | {sum(map(len, r)):,} | {sum(f1s) / len(f1s):.4f} | {exact}/{len(r)} |")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "tokenizer_versions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
