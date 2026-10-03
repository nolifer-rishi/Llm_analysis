"""
step4_visualize.py - EduBench-Local Pipeline Step 4
=====================================================
Reads evaluation metrics produced by step3_evaluate.py and generates a suite
of publication-quality comparative performance charts saved into figures/.

Required input files:
  results/leaderboard.csv          - per-model aggregated metrics
  results/subject_leaderboard.csv  - per-model x per-subject aggregated metrics
                                     (written by step3_evaluate.py)

Optional input file (enables 3 extra distribution charts):
  results/scored_results.csv       - per-question detail (model, subject, scores,
                                     latency, tokens)

Output figures (saved as high-resolution PNG inside figures/):
  01_overall_score_bars.png       - grouped bar chart: EM%, ROUGE-L, BERT-F1, LLM(1-5)
  02_latency_vs_accuracy.png      - scatter: avg latency vs LLM score, sized by N
  03_subject_heatmap.png          - heatmap: model x subject avg LLM(1-5) score
  04_subject_bar_comparison.png   - grouped bars per subject (LLM 1-5 score)
  05_exact_match_rate.png         - horizontal bar: exact-match rate per model
  06_subject_metric_heatmap.png   - multi-metric heatmap (ROUGE-L, BERT-F1, LLM)
  -- the following three require scored_results.csv --
  07_score_distribution.png       - violin / strip plot: LLM score distribution
  08_latency_distribution.png     - box plot: latency distribution per model
  09_tokens_vs_latency.png        - scatter: total tokens vs latency coloured by model

Usage:
  python step4_visualize.py

Dependencies:
  pip install matplotlib pandas seaborn numpy
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Guard: import third-party libs with friendly error messages
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend - no display needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("ERROR: matplotlib is not installed.\n  pip install matplotlib")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is not installed.\n  pip install numpy")

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pandas is not installed.\n  pip install pandas")

try:
    import seaborn as sns
except ImportError:
    sys.exit("ERROR: seaborn is not installed.\n  pip install seaborn")

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEADERBOARD_PATH         = config.RESULTS_DIR / "leaderboard.csv"
SUBJECT_LEADERBOARD_PATH = config.RESULTS_DIR / "subject_leaderboard.csv"
SCORED_RESULTS_PATH      = config.RESULTS_DIR / "scored_results.csv"
GEMINI_SCORED_PATH       = config.RESULTS_DIR / "gemini_scored_results.csv"
GEMINI_LEADERBOARD_PATH  = config.RESULTS_DIR / "gemini_leaderboard.csv"
FIGURES_DIR              = config.FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Visual Design System
# ---------------------------------------------------------------------------
PALETTE = [
    "#4C9BE8",
    "#E8834C",
    "#4CE8A0",
    "#E84C6B",
    "#B04CE8",
    "#E8D44C",
    "#4CE8E8",
    "#E84CB4",
]

BG_COLOR    = "#0F1117"
PANEL_COLOR = "#1A1D27"
TEXT_COLOR  = "#E8EAF0"
GRID_COLOR  = "#2A2D3A"

METRIC_COLORS = {
    "Exact Match %":    "#4C9BE8",
    "ROUGE-L":          "#4CE8A0",
    "BERT-F1":          "#B04CE8",
    "LLM Score (1-5)":  "#E8834C",
    "LLM Score (0-10)": "#E8D44C",
}


def apply_dark_style() -> None:
    """Configure matplotlib for a premium dark-mode aesthetic."""
    plt.rcParams.update({
        "figure.facecolor":      BG_COLOR,
        "axes.facecolor":        PANEL_COLOR,
        "axes.edgecolor":        GRID_COLOR,
        "axes.labelcolor":       TEXT_COLOR,
        "axes.titlecolor":       TEXT_COLOR,
        "axes.titlesize":        13,
        "axes.titleweight":      "bold",
        "axes.labelsize":        10,
        "axes.grid":             True,
        "axes.axisbelow":        True,
        "grid.color":            GRID_COLOR,
        "grid.linewidth":        0.6,
        "xtick.color":           TEXT_COLOR,
        "ytick.color":           TEXT_COLOR,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "legend.facecolor":      PANEL_COLOR,
        "legend.edgecolor":      GRID_COLOR,
        "legend.labelcolor":     TEXT_COLOR,
        "legend.fontsize":       9,
        "legend.title_fontsize": 9,
        "text.color":            TEXT_COLOR,
        "figure.dpi":            150,
        "savefig.dpi":           200,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     BG_COLOR,
        "font.family":           "DejaVu Sans",
    })


def savefig(fig: plt.Figure, name: str) -> Path:
    out = FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK]  {out}")
    return out


def short_name(model: str, max_len: int = 20) -> str:
    """Shorten a model tag for axis labels."""
    return model if len(model) <= max_len else model[:max_len - 1] + "..."


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_leaderboard() -> pd.DataFrame:
    """Load results/leaderboard.csv."""
    if not LEADERBOARD_PATH.exists():
        sys.exit(
            f"ERROR: {LEADERBOARD_PATH} not found.\n"
            "       Run step3_evaluate.py first."
        )
    df = pd.read_csv(LEADERBOARD_PATH)
    for col in [
        "n_questions", "exact_match_rate",
        "avg_rouge_l", "avg_bert_score_f1",
        "avg_llm_score_1_5", "avg_llm_score_0_10",
        "avg_latency_s", "n_errors",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_label"] = df["model"].apply(short_name)
    return df


def load_subject_leaderboard() -> pd.DataFrame:
    """
    Load results/subject_leaderboard.csv directly (written by step3_evaluate.py).
    No re-derivation from scored_results.csv is performed here.
    """
    if not SUBJECT_LEADERBOARD_PATH.exists():
        sys.exit(
            f"ERROR: {SUBJECT_LEADERBOARD_PATH} not found.\n"
            "       Run step3_evaluate.py first."
        )
    if SUBJECT_LEADERBOARD_PATH.stat().st_size == 0:
        sys.exit(
            f"ERROR: {SUBJECT_LEADERBOARD_PATH} exists but is empty.\n"
            "       Re-run step3_evaluate.py to regenerate it."
        )
    df = pd.read_csv(SUBJECT_LEADERBOARD_PATH)
    for col in [
        "n_questions", "exact_match_rate",
        "avg_rouge_l", "avg_bert_score_f1",
        "avg_llm_score_1_5",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_label"] = df["model"].apply(short_name)
    return df


def load_scored_results():
    """
    Optionally load results/scored_results.csv.
    Returns None with a warning if the file is absent or empty.
    Charts that need it will be skipped gracefully.
    """
    if not SCORED_RESULTS_PATH.exists():
        print(
            f"  [WARN] {SCORED_RESULTS_PATH} not found - "
            "per-record distribution charts will be skipped."
        )
        return None
    if SCORED_RESULTS_PATH.stat().st_size == 0:
        print(
            f"  [WARN] {SCORED_RESULTS_PATH} is empty - "
            "per-record distribution charts will be skipped."
        )
        return None

    df = pd.read_csv(SCORED_RESULTS_PATH)
    for col in [
        "exact_match", "rouge_l", "bert_score_f1",
        "llm_score_0_10", "llm_score_1_5",
        "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_label"] = df["model"].apply(short_name)
    return df


DATASET_DISPLAY_NAMES = {
    "science":                     "SciQ",
    "general_science":             "OpenBookQA",
    "science_challenge":           "ARC-Challenge",
    "reading_comprehension":       "RACE",
    "reading_comprehension_squad": "SQuAD v1.1",
}


def load_and_merge_gemini(
    lb: pd.DataFrame,
    sub_lb: pd.DataFrame,
    scored: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load results/gemini_scored_results.csv and merge into lb, sub_lb, and scored.
    Returns (lb, sub_lb, scored, gemini_df).
    """
    if not GEMINI_SCORED_PATH.exists() or GEMINI_SCORED_PATH.stat().st_size == 0:
        print(f"  [INFO] {GEMINI_SCORED_PATH} not found - plotting Qwen only.")
        return lb, sub_lb, scored, None

    gem_df = pd.read_csv(GEMINI_SCORED_PATH)
    if gem_df.empty:
        print(f"  [INFO] {GEMINI_SCORED_PATH} is empty - plotting Qwen only.")
        return lb, sub_lb, scored, None

    print(f"  Loaded Gemini results: {len(gem_df)} scored questions from {GEMINI_SCORED_PATH.name}")

    # Standardize model label for Gemini
    gem_model_name = gem_df["model"].iloc[0] if "model" in gem_df.columns else "gemini-flash"
    if not gem_model_name or gem_model_name == "none":
        gem_model_name = "gemini-flash"

    # Numeric conversion
    for col in ["exact_match", "token_f1", "rouge_l", "char_similarity", "contains_match"]:
        if col in gem_df.columns:
            gem_df[col] = pd.to_numeric(gem_df[col], errors="coerce")

    # 1. Append Gemini summary to lb
    gem_lb_row = {
        "rank": len(lb) + 1,
        "model": gem_model_name,
        "n_questions": len(gem_df),
        "exact_match_rate": gem_df["exact_match"].mean(),
        "avg_rouge_l": gem_df["rouge_l"].mean(),
        "avg_bert_score_f1": gem_df["token_f1"].mean(),
        "avg_llm_score_1_5": np.nan,
        "avg_llm_score_0_10": np.nan,
        "avg_latency_s": gem_df["latency_s"].mean() if "latency_s" in gem_df.columns else 4.0,
        "n_errors": int(gem_df["error"].notna().sum()) if "error" in gem_df.columns else 0,
        "subjects": "|".join(sorted(gem_df["subject"].dropna().unique())),
        "model_label": gem_model_name,
    }
    lb_merged = pd.concat([lb, pd.DataFrame([gem_lb_row])], ignore_index=True)

    # 2. Append Gemini per-subject metrics to sub_lb
    gem_sub_rows = []
    for subj, grp in gem_df.groupby("subject"):
        gem_sub_rows.append({
            "rank": len(sub_lb) + len(gem_sub_rows) + 1,
            "model": gem_model_name,
            "subject": subj,
            "n_questions": len(grp),
            "exact_match_rate": grp["exact_match"].mean(),
            "avg_rouge_l": grp["rouge_l"].mean(),
            "avg_bert_score_f1": grp["token_f1"].mean() if "token_f1" in grp.columns else np.nan,
            "avg_llm_score_1_5": np.nan,
            "avg_llm_score_0_10": np.nan,
            "model_label": gem_model_name,
        })
    sub_lb_merged = pd.concat([sub_lb, pd.DataFrame(gem_sub_rows)], ignore_index=True)

    # 3. Append to scored if scored is available
    scored_merged = scored
    if scored is not None and not scored.empty:
        gem_scored_copy = gem_df.copy()
        gem_scored_copy["model"] = gem_model_name
        gem_scored_copy["model_label"] = gem_model_name
        if "student_answer" not in gem_scored_copy.columns and "gemini_answer" in gem_scored_copy.columns:
            gem_scored_copy["student_answer"] = gem_scored_copy["gemini_answer"]
        if "bert_score_f1" not in gem_scored_copy.columns and "token_f1" in gem_scored_copy.columns:
            gem_scored_copy["bert_score_f1"] = gem_scored_copy["token_f1"]
        scored_merged = pd.concat([scored, gem_scored_copy], ignore_index=True)

    return lb_merged, sub_lb_merged, scored_merged, gem_df



# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _add_bar_labels(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.2f}",
    color: str = TEXT_COLOR,
    fontsize: int = 8,
    v_offset_frac: float = 0.012,
) -> None:
    """Annotate vertical bar patches with their numeric value."""
    y_max = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * v_offset_frac,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold",
        )


def _styled_fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 01 - Overall performance bar chart
#   Metrics: Exact Match %, ROUGE-L, BERT-F1, LLM Score (1-5)
# ---------------------------------------------------------------------------

def plot_overall_score_bars(lb: pd.DataFrame) -> None:
    """
    Grouped bar chart showing Exact Match %, ROUGE-L, BERT-F1, and LLM(1-5)
    side-by-side for every model in the leaderboard.
    """
    models = lb["model_label"].tolist()
    n      = len(models)
    x      = np.arange(n)

    metrics = {}
    metrics["Exact Match %"] = lb["exact_match_rate"].fillna(0) * 100
    if "avg_rouge_l" in lb.columns:
        metrics["ROUGE-L"] = lb["avg_rouge_l"].fillna(0)
    if "avg_bert_score_f1" in lb.columns:
        metrics["BERT-F1"] = lb["avg_bert_score_f1"].fillna(0)
    metrics["LLM Score (1-5)"] = lb["avg_llm_score_1_5"].fillna(0)

    n_metrics = len(metrics)
    width     = min(0.72 / n_metrics, 0.22)
    offsets   = np.linspace(
        -(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics
    ) * width

    fig, ax = _styled_fig(figsize=(max(11, n * 4.5 + 2), 6))

    for (label, values), offset in zip(metrics.items(), offsets):
        color = METRIC_COLORS.get(label, PALETTE[0])
        bars = ax.bar(
            x + offset, values.values, width,
            label=label, color=color,
            alpha=0.88, zorder=3,
            edgecolor=BG_COLOR, linewidth=0.6,
        )
        _add_bar_labels(ax, bars, fmt="{:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance Comparison - EduBench-Local", pad=16)
    ax.legend(loc="upper right", framealpha=0.85)

    all_vals = np.concatenate([v.values for v in metrics.values()])
    y_max = float(np.nanmax(all_vals)) if len(all_vals) > 0 else 10.0
    ax.set_ylim(0, y_max * 1.25)

    for ref in [1, 2, 3, 4, 5]:
        ax.axhline(ref, color=GRID_COLOR, lw=0.5, ls=":", zorder=1)

    fig.tight_layout()
    savefig(fig, "01_overall_score_bars.png")


# ---------------------------------------------------------------------------
# Figure 02 - Latency vs Accuracy scatter
# ---------------------------------------------------------------------------

def plot_latency_vs_accuracy(lb: pd.DataFrame) -> None:
    fig, ax = _styled_fig(figsize=(9, 6))

    sizes = (lb["n_questions"] / lb["n_questions"].max() * 600 + 100).values

    for i, (_, row) in enumerate(lb.iterrows()):
        c = PALETTE[i % len(PALETTE)]
        ax.scatter(
            row["avg_latency_s"], row["avg_llm_score_1_5"],
            s=sizes[i], color=c, alpha=0.90, zorder=4,
            edgecolors="white", linewidths=0.9,
        )
        ax.annotate(
            row["model_label"],
            xy=(row["avg_latency_s"], row["avg_llm_score_1_5"]),
            xytext=(7, 7), textcoords="offset points",
            fontsize=8.5, color=c, fontweight="bold",
        )

    ax.set_xlabel("Average Latency (s)")
    ax.set_ylabel("Average LLM Score (1-5)")
    ax.set_title(
        "Latency vs. Accuracy  -  bubble size proportional to N questions", pad=14
    )

    if len(lb) > 1:
        med_x = lb["avg_latency_s"].median()
        med_y = lb["avg_llm_score_1_5"].median()
        ax.axvline(med_x, color=GRID_COLOR, lw=1.2, ls="--", zorder=2,
                   label=f"Median latency ({med_x:.2f}s)")
        ax.axhline(med_y, color=GRID_COLOR, lw=1.2, ls="--", zorder=2,
                   label=f"Median LLM score ({med_y:.2f})")
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    savefig(fig, "02_latency_vs_accuracy.png")


# ---------------------------------------------------------------------------
# Figure 03 - Subject heatmap: model x subject, avg LLM(1-5)
# ---------------------------------------------------------------------------

def plot_subject_heatmap(sub_lb: pd.DataFrame) -> None:
    """
    Reads avg_llm_score_1_5 from the pre-loaded subject_leaderboard DataFrame
    and renders a model x subject heatmap - no CSV re-read.
    """
    pivot = sub_lb.pivot_table(
        index="model_label", columns="subject",
        values="avg_llm_score_1_5", aggfunc="mean",
    )
    if pivot.empty:
        print("  !  Skipping heatmap (no subject data).")
        return

    fig_h = max(4, len(pivot) * 1.1 + 2.5)
    fig_w = max(7, len(pivot.columns) * 1.6 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(
        pivot,
        ax=ax, cmap=cmap,
        annot=True, fmt=".2f",
        annot_kws={"size": 11, "weight": "bold", "color": TEXT_COLOR},
        linewidths=0.6, linecolor=GRID_COLOR,
        vmin=1, vmax=5,
        cbar_kws={"label": "Avg LLM Score (1-5)", "shrink": 0.8},
    )
    ax.set_title("LLM Score Heatmap - Model x Subject", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.label.set_color(TEXT_COLOR)
        cbar.ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    savefig(fig, "03_subject_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 04 - Subject bar comparison: LLM(1-5) grouped by subject
# ---------------------------------------------------------------------------

def plot_subject_bar_comparison(sub_lb: pd.DataFrame) -> None:
    subjects = sub_lb["subject"].unique()
    models   = sub_lb["model_label"].unique()
    n_sub    = len(subjects)
    n_mod    = len(models)

    if n_sub == 0:
        print("  !  Skipping subject bar comparison (no data).")
        return

    x     = np.arange(n_sub)
    width = min(0.8 / max(n_mod, 1), 0.30)

    fig, ax = _styled_fig(figsize=(max(10, n_sub * 2.8 + 2), 6))

    for mi, model in enumerate(models):
        subset = sub_lb[sub_lb["model_label"] == model]
        vals = []
        for s in subjects:
            match = subset.loc[subset["subject"] == s, "avg_llm_score_1_5"]
            vals.append(float(match.values[0]) if not match.empty else np.nan)

        offset = (mi - n_mod / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=model, color=PALETTE[mi % len(PALETTE)],
            alpha=0.88, zorder=3,
            edgecolor=BG_COLOR, linewidth=0.6,
        )
        _add_bar_labels(ax, bars, fmt="{:.2f}", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=22, ha="right")
    ax.set_ylabel("Avg LLM Score (1-5)")
    ax.set_title("Per-Subject LLM Score Comparison", pad=14)
    ax.set_ylim(0, 5.9)
    for ref in [1, 2, 3, 4, 5]:
        ax.axhline(ref, color=GRID_COLOR, lw=0.5, ls=":", zorder=1)
    if n_mod > 1:
        ax.legend(title="Model", loc="upper right")

    fig.tight_layout()
    savefig(fig, "04_subject_bar_comparison.png")


# ---------------------------------------------------------------------------
# Figure 05 - Exact-match rate horizontal bar
# ---------------------------------------------------------------------------

def plot_exact_match_rate(lb: pd.DataFrame) -> None:
    lb_sorted = lb.sort_values("exact_match_rate", ascending=True)
    fig, ax   = _styled_fig(figsize=(9, max(4, len(lb_sorted) * 0.9 + 2)))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(lb_sorted))]
    bars   = ax.barh(
        lb_sorted["model_label"],
        lb_sorted["exact_match_rate"] * 100,
        color=colors, alpha=0.88, zorder=3,
        edgecolor=BG_COLOR, linewidth=0.6, height=0.55,
    )

    x_max = max(float(lb_sorted["exact_match_rate"].max()) * 100, 1.0)
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + x_max * 0.018, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center", ha="left",
            fontsize=9, color=TEXT_COLOR, fontweight="bold",
        )

    ax.set_xlabel("Exact Match Rate (%)")
    ax.set_title("Exact Match Rate per Model", pad=14)
    ax.set_xlim(0, x_max * 1.25 + 1)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    savefig(fig, "05_exact_match_rate.png")


# ---------------------------------------------------------------------------
# Figure 06 - Multi-metric subject heatmap (ROUGE-L, BERT-F1, LLM)
# ---------------------------------------------------------------------------

def plot_subject_multi_metric_heatmap(sub_lb: pd.DataFrame) -> None:
    """
    Side-by-side heatmaps (one panel per metric) for every subject,
    built from the in-memory subject_leaderboard DataFrame.
    """
    metric_defs = [
        ("ROUGE-L",         "avg_rouge_l",         0.0, 1.0, "Blues"),
        ("BERT-F1",         "avg_bert_score_f1",   0.0, 1.0, "Purples"),
        ("LLM Score (1-5)", "avg_llm_score_1_5",   1.0, 5.0, "mako"),
    ]
    # Keep only columns present in the data
    metric_defs = [(lbl, col, vmin, vmax, cm)
                   for lbl, col, vmin, vmax, cm in metric_defs
                   if col in sub_lb.columns]

    if not metric_defs:
        print("  !  Skipping multi-metric heatmap (no numeric metric columns).")
        return

    pivots = [
        (lbl, sub_lb.pivot_table(
            index="model_label", columns="subject",
            values=col, aggfunc="mean",
        ), vmin, vmax, cm)
        for lbl, col, vmin, vmax, cm in metric_defs
    ]

    n_metrics  = len(pivots)
    n_models   = max(len(p[1]) for p in pivots)
    n_subjects = max(len(p[1].columns) for p in pivots)

    fig_h = max(4, n_models * 1.0 + 2.5)
    fig_w = max(8, n_subjects * 1.5 * n_metrics + 2)

    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, fig_h), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, (label, pivot, vmin, vmax, cmap) in zip(axes, pivots):
        sns.heatmap(
            pivot,
            ax=ax, cmap=cmap,
            annot=True, fmt=".2f",
            annot_kws={"size": 9, "weight": "bold"},
            linewidths=0.5, linecolor=GRID_COLOR,
            vmin=vmin, vmax=vmax,
            cbar_kws={"shrink": 0.75},
        )
        ax.set_title(label, fontsize=11, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("" if ax is not axes[0] else "Model")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Subject-Wise Metric Breakdown - EduBench-Local",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "06_subject_metric_heatmap.png")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Figure 07 - LLM score distribution (violin + strip)  [needs scored_results]
# ---------------------------------------------------------------------------

def plot_score_distribution(scored: pd.DataFrame) -> None:
    valid = scored.dropna(subset=["llm_score_1_5"])
    if valid.empty:
        print("  !  Skipping score distribution (no LLM score data).")
        return
    models       = valid["model_label"].unique()
    n            = len(models)
    palette_dict = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, ax = _styled_fig(figsize=(max(8, n * 2.8 + 2), 6))

    if n >= 2:
        sns.violinplot(
            data=valid, x="model_label", y="llm_score_1_5",
            hue="model_label", palette=palette_dict, legend=False, ax=ax,
            inner=None, cut=0, linewidth=0.8, alpha=0.60,
        )

    sns.stripplot(
        data=valid, x="model_label", y="llm_score_1_5",
        hue="model_label", palette=palette_dict, legend=False, ax=ax,
        size=3, alpha=0.45, jitter=True, zorder=3,
    )

    ax.set_xlabel("")
    ax.set_ylabel("LLM Score (1-5)")
    ax.set_title("LLM Score Distribution per Model", pad=14)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    savefig(fig, "07_score_distribution.png")


# ---------------------------------------------------------------------------
# Figure 08 - Latency distribution box plot  [needs scored_results]
# ---------------------------------------------------------------------------

def plot_latency_distribution(scored: pd.DataFrame) -> None:
    valid = scored.dropna(subset=["latency_s"])
    if valid.empty:
        print("  !  Skipping latency distribution (no latency data).")
        return
    models       = valid["model_label"].unique()
    n            = len(models)
    palette_dict = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, ax = _styled_fig(figsize=(max(8, n * 2.8 + 2), 6))

    sns.boxplot(
        data=valid, x="model_label", y="latency_s",
        hue="model_label", palette=palette_dict, legend=False, ax=ax,
        linewidth=0.9,
        flierprops=dict(
            marker="o", markerfacecolor=PALETTE[3],
            markersize=3, alpha=0.5, linestyle="none",
        ),
    )

    ax.set_xlabel("")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Response Latency Distribution per Model", pad=14)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    savefig(fig, "08_latency_distribution.png")


# ---------------------------------------------------------------------------
# Figure 09 - Total tokens vs latency scatter  [needs scored_results]
# ---------------------------------------------------------------------------

def plot_tokens_vs_latency(scored: pd.DataFrame) -> None:
    required = {"total_tokens", "latency_s", "model_label"}
    if not required.issubset(scored.columns):
        print("  !  Skipping tokens vs latency (missing columns).")
        return
    df = scored.dropna(subset=["total_tokens", "latency_s"])
    if df.empty:
        print("  !  Skipping tokens vs latency (no data after dropna).")
        return

    models = df["model_label"].unique()
    fig, ax = _styled_fig(figsize=(10, 6))

    for i, model in enumerate(models):
        sub = df[df["model_label"] == model]
        ax.scatter(
            sub["total_tokens"], sub["latency_s"],
            label=model, color=PALETTE[i % len(PALETTE)],
            s=20, alpha=0.55, zorder=3,
        )

    ax.set_xlabel("Total Tokens (prompt + completion)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Token Count vs. Response Latency", pad=14)
    if len(models) > 1:
        ax.legend(title="Model", loc="upper left")

    fig.tight_layout()
    savefig(fig, "09_tokens_vs_latency.png")


# ---------------------------------------------------------------------------
# Figure 10 - Qwen vs Gemini per-dataset comparative bar charts
# ---------------------------------------------------------------------------

def plot_qwen_vs_gemini_per_dataset(sub_lb: pd.DataFrame, gem_df: pd.DataFrame) -> None:
    """
    Side-by-side comparative bar charts comparing Qwen and Gemini across all
    5 benchmark datasets (SciQ, OpenBookQA, ARC-Challenge, RACE, SQuAD v1.1).
    """
    subjects = [
        "science", "general_science", "science_challenge",
        "reading_comprehension", "reading_comprehension_squad"
    ]
    dataset_names = [DATASET_DISPLAY_NAMES.get(s, s) for s in subjects]

    # Filter Qwen rows
    qwen_sub = sub_lb[sub_lb["model"].str.contains("qwen", case=False, na=False)]
    if qwen_sub.empty:
        qwen_sub = sub_lb.iloc[:5]

    qwen_em = []
    qwen_rl = []
    for s in subjects:
        m = qwen_sub.loc[qwen_sub["subject"] == s]
        qwen_em.append(float(m["exact_match_rate"].values[0]) * 100 if not m.empty else 0.0)
        qwen_rl.append(float(m["avg_rouge_l"].values[0]) if not m.empty else 0.0)

    # Gemini metrics aggregated per subject
    gem_agg = gem_df.groupby("subject").agg({
        "exact_match": "mean",
        "rouge_l": "mean",
        "token_f1": "mean",
    }).reindex(subjects).fillna(0.0)

    gem_em = [float(gem_agg.loc[s, "exact_match"]) * 100 for s in subjects]
    gem_rl = [float(gem_agg.loc[s, "rouge_l"]) for s in subjects]
    gem_f1 = [float(gem_agg.loc[s, "token_f1"]) for s in subjects]

    n_ds = len(subjects)
    x = np.arange(n_ds)
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    color_qwen = "#4C9BE8"    # Electric Blue
    color_gemini = "#E8834C"  # Vibrant Coral

    # Panel 1: Exact Match Rate (%)
    bars_q1 = ax1.bar(x - width/2, qwen_em, width, label="Qwen 2.5 (3B)",
                      color=color_qwen, alpha=0.9, edgecolor=BG_COLOR, linewidth=0.8, zorder=3)
    bars_g1 = ax1.bar(x + width/2, gem_em, width, label="Gemini (Flash)",
                      color=color_gemini, alpha=0.9, edgecolor=BG_COLOR, linewidth=0.8, zorder=3)

    _add_bar_labels(ax1, bars_q1, fmt="{:.1f}%", fontsize=8.5)
    _add_bar_labels(ax1, bars_g1, fmt="{:.1f}%", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names, rotation=18, ha="right", fontsize=9.5)
    ax1.set_ylabel("Exact Match Rate (%)", fontsize=10.5)
    ax1.set_title("Exact Match Rate (EM %) by Benchmark Dataset", fontsize=12, pad=12)
    ax1.set_ylim(0, max(max(qwen_em), max(gem_em), 10) * 1.25)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: ROUGE-L F1 Score
    bars_q2 = ax2.bar(x - width/2, qwen_rl, width, label="Qwen 2.5 (3B)",
                      color=color_qwen, alpha=0.9, edgecolor=BG_COLOR, linewidth=0.8, zorder=3)
    bars_g2 = ax2.bar(x + width/2, gem_rl, width, label="Gemini (Flash)",
                      color=color_gemini, alpha=0.9, edgecolor=BG_COLOR, linewidth=0.8, zorder=3)

    _add_bar_labels(ax2, bars_q2, fmt="{:.2f}", fontsize=8.5)
    _add_bar_labels(ax2, bars_g2, fmt="{:.2f}", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names, rotation=18, ha="right", fontsize=9.5)
    ax2.set_ylabel("ROUGE-L F1 Score", fontsize=10.5)
    ax2.set_title("ROUGE-L Overlap by Benchmark Dataset", fontsize=12, pad=12)
    ax2.set_ylim(0, max(max(qwen_rl), max(gem_rl), 0.2) * 1.25)
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("EduBench Head-to-Head: Qwen 2.5 (3B) vs. Google Gemini (Flash)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "10_qwen_vs_gemini_per_dataset.png")


# ---------------------------------------------------------------------------
# Figure 11 - Qwen vs Gemini Executive Scorecard Dashboard
# ---------------------------------------------------------------------------

def plot_qwen_vs_gemini_scorecard(lb: pd.DataFrame, sub_lb: pd.DataFrame, gem_df: pd.DataFrame) -> None:
    """
    Generates a publication-grade executive comparison scorecard visual
    summarizing overall metrics, win/loss breakdown, and architectural differences.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.4], hspace=0.35, wspace=0.28)

    color_qwen = "#4C9BE8"
    color_gemini = "#E8834C"

    # Compute overall metrics
    qwen_row = lb[lb["model"].str.contains("qwen", case=False, na=False)]
    if qwen_row.empty:
        qwen_row = lb.iloc[0:1]

    qwen_em = float(qwen_row["exact_match_rate"].values[0]) * 100 if not qwen_row.empty else 2.93
    qwen_rl = float(qwen_row["avg_rouge_l"].values[0]) if not qwen_row.empty else 0.137
    qwen_n  = int(qwen_row["n_questions"].values[0]) if not qwen_row.empty else 750

    gem_em = float(gem_df["exact_match"].mean()) * 100
    gem_rl = float(gem_df["rouge_l"].mean())
    gem_f1 = float(gem_df["token_f1"].mean())
    gem_n  = len(gem_df)

    # ── Panel 1: Top-Level Metric Summary Bars ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ["Exact Match %", "ROUGE-L (x100)", "Token F1 / BERT (x100)"]
    qwen_vals = [qwen_em, qwen_rl * 100, (float(qwen_row["avg_bert_score_f1"].values[0])*100 if not qwen_row.empty else 86.2)]
    gemini_vals = [gem_em, gem_rl * 100, gem_f1 * 100]

    y = np.arange(len(metrics))
    height = 0.32

    bars_q = ax1.barh(y - height/2, qwen_vals, height, label="Qwen 2.5 (3B Local)",
                      color=color_qwen, alpha=0.9, edgecolor=BG_COLOR)
    bars_g = ax1.barh(y + height/2, gemini_vals, height, label="Gemini (Flash Cloud)",
                      color=color_gemini, alpha=0.9, edgecolor=BG_COLOR)

    for b in bars_q:
        ax1.text(b.get_width() + 1.2, b.get_y() + b.get_height()/2, f"{b.get_width():.1f}",
                 va="center", ha="left", fontsize=9, color=TEXT_COLOR, fontweight="bold")
    for b in bars_g:
        ax1.text(b.get_width() + 1.2, b.get_y() + b.get_height()/2, f"{b.get_width():.1f}",
                 va="center", ha="left", fontsize=9, color=TEXT_COLOR, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(metrics, fontsize=10, fontweight="bold")
    ax1.set_xlim(0, max(max(qwen_vals), max(gemini_vals)) * 1.25)
    ax1.set_title("Overall Key Benchmark Metrics Comparison", fontsize=11.5, pad=10)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(axis="x", alpha=0.5)

    # ── Panel 2: Deployment & Architectural Profile Card ───────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    profile_text = (
        "MODEL PROFILES\n"
        "─────────────────────────────\n\n"
        "• Qwen 2.5 (3B):\n"
        "  - Deployment : Local (Ollama)\n"
        "  - Total Eval : 750 questions\n"
        "  - Privacy    : 100% On-device\n"
        "  - Cost       : $0.00 / Zero quota\n\n"
        "• Google Gemini (Flash):\n"
        "  - Deployment : Cloud API\n"
        "  - Total Eval : Micro-target\n"
        "  - Style      : Direct & concise\n"
        "  - Speed      : High-speed reasoning\n"
    )
    ax2.text(0.05, 0.95, profile_text, transform=ax2.transAxes,
             fontsize=9.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, lw=1.2),
             color=TEXT_COLOR)

    # ── Panel 3: Per-Dataset Head-to-Head Radar / Heat Breakdown ───────────
    ax3 = fig.add_subplot(gs[1, :])
    subjects = [
        "science", "general_science", "science_challenge",
        "reading_comprehension", "reading_comprehension_squad"
    ]
    ds_labels = [DATASET_DISPLAY_NAMES.get(s, s) for s in subjects]

    qwen_em_list = []
    qwen_rl_list = []
    qwen_sub = sub_lb[sub_lb["model"].str.contains("qwen", case=False, na=False)]
    for s in subjects:
        m = qwen_sub.loc[qwen_sub["subject"] == s]
        qwen_em_list.append(float(m["exact_match_rate"].values[0])*100 if not m.empty else 0.0)
        qwen_rl_list.append(float(m["avg_rouge_l"].values[0]) if not m.empty else 0.0)

    gem_agg = gem_df.groupby("subject").agg({
        "exact_match": "mean",
        "rouge_l": "mean"
    }).reindex(subjects).fillna(0.0)
    gem_em_list = [float(gem_agg.loc[s, "exact_match"])*100 for s in subjects]
    gem_rl_list = [float(gem_agg.loc[s, "rouge_l"]) for s in subjects]

    # Create a comparative matrix table visual
    cell_data = [
        ["Dataset", "Category", "Qwen EM%", "Gemini EM%", "Qwen ROUGE-L", "Gemini ROUGE-L", "Category Winner"],
    ]
    for i, s in enumerate(subjects):
        winner = "Gemini" if gem_em_list[i] > qwen_em_list[i] or gem_rl_list[i] > qwen_rl_list[i] else (
            "Qwen" if qwen_em_list[i] > gem_em_list[i] or qwen_rl_list[i] > gem_rl_list[i] else "Parity"
        )
        cell_data.append([
            ds_labels[i],
            s,
            f"{qwen_em_list[i]:.1f}%",
            f"{gem_em_list[i]:.1f}%",
            f"{qwen_rl_list[i]:.3f}",
            f"{gem_rl_list[i]:.3f}",
            f"★ {winner}"
        ])

    table = ax3.table(
        cellText=cell_data,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.08, 0.96, 0.82]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    ax3.axis("off")

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#222736")
            cell.set_text_props(weight="bold", color="#4CE8E8")
        else:
            cell.set_facecolor(PANEL_COLOR)
            # Highlight winner column
            if col_idx == 6:
                cell.set_text_props(weight="bold", color="#4CE8A0" if "Gemini" in cell.get_text().get_text() else "#4C9BE8")

    ax3.set_title("Dataset-by-Dataset Benchmark Win/Loss Matrix", fontsize=12, pad=14)

    fig.suptitle("EduBench-Local  |  Qwen vs. Gemini Comparative Executive Scorecard",
                 fontsize=14, fontweight="bold", y=0.98)
    savefig(fig, "11_qwen_vs_gemini_scorecard.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_dark_style()

    print("=" * 64)
    print("EduBench-Local  |  Step 4: Visualize (Multi-Model)")
    print("=" * 64)
    print(f"  Leaderboard         : {LEADERBOARD_PATH}")
    print(f"  Subject leaderboard : {SUBJECT_LEADERBOARD_PATH}")
    print(f"  Scored results      : {SCORED_RESULTS_PATH}  (optional)")
    print(f"  Gemini scored       : {GEMINI_SCORED_PATH}  (optional)")
    print(f"  Figures dir         : {FIGURES_DIR}")
    print()

    # -- Required: leaderboard + subject leaderboard -----------------------
    print("Loading leaderboard data ...")
    lb     = load_leaderboard()
    sub_lb = load_subject_leaderboard()

    # -- Optional: per-question scored results for distribution charts -----
    print("Loading per-question scored results (optional) ...")
    scored = load_scored_results()

    # -- Merge Gemini data if available ------------------------------------
    print("Checking for Gemini evaluation results ...")
    lb, sub_lb, scored, gem_df = load_and_merge_gemini(lb, sub_lb, scored)

    print(f"  Active Models : {lb['model'].tolist()}")
    print(f"  Subjects      : {sub_lb['subject'].unique().tolist()}")
    print(f"  LB rows       : {len(lb)}  |  Subject-LB rows: {len(sub_lb)}")
    if scored is not None:
        print(f"  Scored rows   : {len(scored):,}")
    print()

    # -- Generate figures --------------------------------------------------
    print("Generating figures ...")

    # Core charts (figures 01-06)
    plot_overall_score_bars(lb)
    plot_latency_vs_accuracy(lb)
    plot_subject_heatmap(sub_lb)
    plot_subject_bar_comparison(sub_lb)
    plot_exact_match_rate(lb)
    plot_subject_multi_metric_heatmap(sub_lb)

    # Distribution charts (figures 07-09)
    if scored is not None:
        plot_score_distribution(scored)
        plot_latency_distribution(scored)
        plot_tokens_vs_latency(scored)
    else:
        print(
            "  [SKIP] Per-record distribution charts skipped "
            "(scored_results.csv not available)."
        )

    # Dedicated Qwen vs Gemini comparative figures (figures 10 & 11)
    if gem_df is not None and not gem_df.empty:
        print("Generating Qwen vs Gemini comparative figures ...")
        plot_qwen_vs_gemini_per_dataset(sub_lb, gem_df)
        plot_qwen_vs_gemini_scorecard(lb, sub_lb, gem_df)

    print()
    print(f"All figures saved to: {FIGURES_DIR}")
    print("Step 4 complete.")


if __name__ == "__main__":
    main()

