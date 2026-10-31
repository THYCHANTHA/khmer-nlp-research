"""Render the static report figures (PNG) from reports/experiments with Plotly + headless Microsoft Edge.

Khmer script needs a complex-text shaper; a browser renders it correctly, plain Matplotlib may not. Requires Edge.
    python scripts/render_figures.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from khmer_nlp.viz import figures as fg

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
OUT = ROOT / "reports" / "figures"
EDGE = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def screenshot(fig, name: str, width: int = 1200) -> None:
    exe = next((p for p in EDGE if p.exists()), None)
    if exe is None:
        raise SystemExit("Microsoft Edge not found")
    OUT.mkdir(parents=True, exist_ok=True)
    height = int(fig.layout.height or 450) + 20
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "f.html"
        fig.update_layout(width=width - 40)
        html.write_text(
            "<html><head><meta charset='utf-8'></head><body style='margin:0;background:white'>"
            + fig.to_html(full_html=False, include_plotlyjs=True, config={"staticPlot": True})
            + "</body></html>",
            encoding="utf-8",
        )
        target = OUT / f"{name}.png"
        subprocess.run(
            [
                str(exe),
                "--headless=new",
                "--disable-gpu",
                f"--window-size={width},{height}",
                "--hide-scrollbars",
                "--virtual-time-budget=6000",
                f"--screenshot={target}",
                html.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    print("wrote", target.relative_to(ROOT))


def main() -> None:
    stats = pd.read_csv(EXP / "tfidf_stats.csv")
    kw = pd.read_csv(EXP / "tfidf_keywords.csv")
    top = pd.read_csv(EXP / "tfidf_top_terms.csv")
    rf = pd.read_csv(EXP / "tfidf_rank_frequency.csv")
    summ = pd.read_csv(EXP / "ir_proxy_summary.csv")
    paired = pd.read_csv(EXP / "ir_proxy_paired_vs_A.csv")
    cls = pd.read_csv(EXP / "classification_summary.csv")
    prof = json.loads((EXP / "removal_profiles.json").read_text(encoding="utf-8"))
    names = list(stats.strategy)

    screenshot(fg.token_reduction(stats), "token_reduction")
    screenshot(fg.vocabulary(stats), "tokens_and_vocabulary")
    screenshot(fg.concentration(stats), "concentration")
    screenshot(fg.top_keywords(kw, names[:3], 20), "top20_keywords", 1500)
    screenshot(fg.top_keywords(kw, names[:3], 50), "top50_keywords", 1500)
    screenshot(fg.rank_frequency(rf), "rank_frequency")
    screenshot(fg.before_after_terms(top, names[0], names[1], 15), "frequent_words_before_after")
    first = next(n for n in prof)
    screenshot(fg.removed_words(prof[first], f"Most frequently removed words — {first}"), "most_removed_words")
    screenshot(fg.ir_metrics(summ), "ir_proxy_metrics")
    screenshot(fg.ir_paired(paired), "ir_proxy_paired")
    screenshot(fg.classification(cls), "classification")


if __name__ == "__main__":
    main()
