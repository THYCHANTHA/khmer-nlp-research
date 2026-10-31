"""Plotly figures shared by the Streamlit app and the static figure renderer. They take result tables as input; no
numbers are typed in here. White background, dark-navy text, light-blue accents."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY = "#12294d"
BLUE = "#2f80c9"
LIGHT = "#8cc4ee"
GREY = "#9aa8bb"
PALE = "#e8f1fb"
FONT = "Khmer UI, Noto Sans Khmer, Leelawadee UI, Segoe UI, sans-serif"
# one lightness-separated colour per strategy so the figures also read in greyscale
COLORS = [GREY, LIGHT, BLUE, "#d9a441"]


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": NAVY}},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": FONT, "color": NAVY, "size": 13},
        height=height,
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False, linecolor=GREY)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False, linecolor=GREY)
    return fig


def token_reduction(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=stats["strategy"],
            y=stats["token_reduction_pct"],
            marker_color=[_color(i) for i in range(len(stats))],
            text=[f"{v:.1f}%" for v in stats["token_reduction_pct"]],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="tokens removed (%)", range=[0, max(stats["token_reduction_pct"].max() * 1.2, 1)])
    return _layout(fig, "Token reduction by strategy")


def vocabulary(stats: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tokens", "Vocabulary (distinct terms)"))
    for j, col in enumerate(("tokens", "vocab"), start=1):
        fig.add_trace(
            go.Bar(
                x=stats["strategy"],
                y=stats[col],
                marker_color=[_color(i) for i in range(len(stats))],
                text=[f"{v:,.0f}" for v in stats[col]],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=j,
        )
    return _layout(fig, "Tokens fall sharply, vocabulary barely changes", 420)


def concentration(stats: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Share of tokens in the 10 commonest terms", "Normalised entropy")
    )
    for j, col in enumerate(("top10_token_share", "norm_entropy"), start=1):
        fig.add_trace(
            go.Bar(
                x=stats["strategy"],
                y=stats[col],
                marker_color=[_color(i) for i in range(len(stats))],
                text=[f"{v:.3f}" for v in stats[col]],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=j,
        )
    return _layout(fig, "Information concentration of the term distribution", 420)


def top_keywords(kw: pd.DataFrame, strategies: list[str], k: int = 20) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(strategies), subplot_titles=strategies, horizontal_spacing=0.12)
    for j, s in enumerate(strategies, start=1):
        d = kw[(kw.strategy == s) & (kw["rank"] <= k)].sort_values("rank", ascending=False)
        fig.add_trace(
            go.Bar(x=d["mean_tfidf"], y=d["term"], orientation="h", marker_color=_color(j - 1), showlegend=False),
            row=1,
            col=j,
        )
    fig.update_xaxes(title="mean TF-IDF")
    return _layout(fig, f"Top {k} keywords by mean TF-IDF", 150 + 24 * k)


def rank_frequency(rf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (s, d) in enumerate(rf.groupby("strategy", sort=False)):
        fig.add_trace(go.Scatter(x=d["rank"], y=d["tf"], mode="lines", name=s, line={"color": _color(i), "width": 2.5}))
    fig.update_xaxes(type="log", title="term rank")
    fig.update_yaxes(type="log", title="term frequency")
    return _layout(fig, "Word-frequency distribution (log-log)", 420)


def before_after_terms(top_terms: pd.DataFrame, a: str, b: str, k: int = 15) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{a}", f"{b}"), horizontal_spacing=0.15)
    for j, (s, color) in enumerate(((a, GREY), (b, LIGHT)), start=1):
        d = top_terms[(top_terms.strategy == s) & (top_terms["rank"] <= k)].sort_values("rank", ascending=False)
        fig.add_trace(
            go.Bar(x=d["tf"], y=d["term"], orientation="h", marker_color=color, showlegend=False), row=1, col=j
        )
    fig.update_xaxes(title="term frequency")
    return _layout(fig, "Most frequent words before and after removal", 150 + 24 * k)


def removed_words(profile: dict, title: str) -> go.Figure:
    d = pd.DataFrame(profile["top_removed"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["count"], y=d["word"], orientation="h", marker_color=BLUE))
    fig.update_xaxes(title="occurrences removed")
    return _layout(fig, title, 150 + 18 * len(d))


def ir_metrics(summary: pd.DataFrame, metrics: tuple[str, ...] = ("P@10", "R@10", "MRR", "nDCG@10")) -> go.Figure:
    fig = go.Figure()
    for i in range(len(summary)):
        vals = [summary.iloc[i][m] for m in metrics]
        lo = [summary.iloc[i][f"{m}_lo"] for m in metrics]
        hi = [summary.iloc[i][f"{m}_hi"] for m in metrics]
        fig.add_trace(
            go.Bar(
                name=summary.iloc[i]["strategy"],
                x=list(metrics),
                y=vals,
                marker_color=_color(i),
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [h - v for h, v in zip(hi, vals, strict=True)],
                    "arrayminus": [v - low for v, low in zip(vals, lo, strict=True)],
                    "color": NAVY,
                },
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="mean over queries (95% bootstrap CI)", rangemode="tozero")
    return _layout(fig, "Exploratory proxy retrieval metrics", 440)


def ir_paired(paired: pd.DataFrame, metrics: tuple[str, ...] = ("P@10", "R@10", "MRR", "nDCG@10")) -> go.Figure:
    d = paired[paired.metric.isin(metrics)]
    fig = go.Figure()
    for i, (s, g) in enumerate(d.groupby("strategy", sort=False), start=1):
        fig.add_trace(
            go.Scatter(
                x=g["diff_vs_A"],
                y=g["metric"],
                mode="markers",
                name=s,
                marker={"color": _color(i), "size": 11},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": (g["hi"] - g["diff_vs_A"]).tolist(),
                    "arrayminus": (g["diff_vs_A"] - g["lo"]).tolist(),
                    "color": _color(i),
                },
            )
        )
    fig.add_vline(x=0, line_color=NAVY, line_dash="dash")
    fig.update_xaxes(title="difference vs. no removal (paired, 95% bootstrap CI)")
    return _layout(fig, "Change relative to Strategy A", 400)


def classification(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=summary["strategy"],
            y=summary["macro_f1_mean"],
            error_y={"type": "data", "array": summary["macro_f1_sd"].fillna(0)},
            marker_color=[_color(i) for i in range(len(summary))],
            text=[f"{v:.3f}" for v in summary["macro_f1_mean"]],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="macro-F1 (mean ± sd over folds)", range=[0, 1.1])
    return _layout(fig, "TF-IDF + Linear SVC, Wikipedia-category prediction", 420)


def dictionary_quality(report: dict) -> go.Figure:
    items = [
        ("raw rows", report["raw_rows"]),
        ("unique raw", report["unique_raw"]),
        ("unique normalised", report["unique_normalized"]),
        ("duplicate entries", report["duplicate_entries"]),
        ("padded with spaces", report["padded_entries"]),
        ("empty", report["empty_entries"]),
        ("multi-word", report["multiword_entries"]),
        ("conflicting labels", report["conflicting_terms"]),
    ]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color=BLUE))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Dictionary quality (imported file)", 380)


def category_distribution(counts: dict[str, int]) -> go.Figure:
    d = pd.Series(counts).sort_values()
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker_color=LIGHT))
    fig.update_xaxes(title="dictionary entries")
    return _layout(fig, "Category distribution of the imported dictionary", 150 + 24 * len(d))


def retained(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("count")
    fig = go.Figure(go.Bar(x=d["count"], y=d["word"], orientation="h", marker_color="#d9a441"))
    fig.update_xaxes(title="occurrences kept")
    return _layout(fig, "Protected words kept by the strategy (negations etc.)", 150 + 22 * len(d))
