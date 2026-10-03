"""
EduBench-Local: Visualizer
============================
Generates publication-quality figures for the research paper.
All figures saved to results/figures/ as PNG (300 DPI).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Use non-interactive backend for saving
matplotlib.use("Agg")

# ── Style Configuration ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
})

# Color palette
PALETTE = sns.color_palette("husl", 8)
METRIC_COLORS = {
    "ROUGE-L": "#2ecc71",
    "BERTScore": "#3498db",
    "Judge Score": "#e74c3c",
}


def load_data():
    """Load scored results and leaderboard."""
    df = pd.read_csv(config.SCORED_RESULTS_FILE)
    leaderboard = pd.read_csv(config.LEADERBOARD_FILE) if config.LEADERBOARD_FILE.exists() else None
    return df, leaderboard


def fig1_dataset_metric_bars(df, save=True):
    """
    Bar chart: Average scores per dataset, grouped by metric.
    This is the primary results figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    
    metrics = [
        ("rouge_l", "ROUGE-L F1", METRIC_COLORS["ROUGE-L"]),
        ("bertscore_f1", "BERTScore F1", METRIC_COLORS["BERTScore"]),
        ("judge_score", "Judge Score (1-5)", METRIC_COLORS["Judge Score"]),
    ]
    
    for ax, (col, title, color) in zip(axes, metrics):
        means = df.groupby("dataset")[col].mean().sort_values(ascending=False)
        stds = df.groupby("dataset")[col].std()
        
        bars = ax.barh(
            range(len(means)), means.values,
            xerr=stds[means.index].values,
            color=color, alpha=0.8, edgecolor="white", linewidth=0.5,
            capsize=3,
        )
        ax.set_yticks(range(len(means)))
        ax.set_yticklabels(means.index)
        ax.set_xlabel(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(title, fontweight="bold")
    
    fig.suptitle(f"{config.MODEL_DISPLAY_NAME} Performance Across Educational Datasets", fontweight="bold", fontsize=14)
    plt.tight_layout()
    
    if save:
        path = config.FIGURES_DIR / "fig1_dataset_metric_bars.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    
    plt.close(fig)
    return fig


def fig2_radar_chart(df, save=True):
    """
    Radar chart: Mistral 7B's performance profile across datasets.
    """
    datasets = df["dataset"].unique()
    
    # Compute normalized averages per dataset
    metrics = ["rouge_l", "bertscore_f1"]
    # Normalize judge_score to 0-1 (it's 1-5)
    df_norm = df.copy()
    df_norm["judge_norm"] = (df_norm["judge_score"] - 1) / 4
    metrics_norm = ["rouge_l", "bertscore_f1", "judge_norm"]
    metric_labels = ["ROUGE-L", "BERTScore", "Judge"]
    
    means = df_norm.groupby("dataset")[metrics_norm].mean()
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = sns.color_palette("husl", len(datasets))
    
    for i, dataset in enumerate(means.index):
        values = means.loc[dataset].values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=dataset, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"{config.MODEL_DISPLAY_NAME} — Multi-Metric Profile by Dataset", pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    if save:
        path = config.FIGURES_DIR / "fig2_radar_chart.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    
    plt.close(fig)
    return fig


def fig3_heatmap(df, save=True):
    """
    Heatmap: Dataset × Metric matrix showing average scores.
    """
    metrics = ["rouge_l", "bertscore_f1", "judge_score"]
    metric_labels = ["ROUGE-L", "BERTScore F1", "Judge Score (1-5)"]
    
    pivot = df.groupby("dataset")[metrics].mean()
    pivot.columns = metric_labels
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Score"},
    )
    
    ax.set_title(f"{config.MODEL_DISPLAY_NAME} — Score Heatmap (Dataset × Metric)", fontweight="bold")
    ax.set_ylabel("Dataset")
    ax.set_xlabel("Metric")
    
    if save:
        path = config.FIGURES_DIR / "fig3_heatmap.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    
    plt.close(fig)
    return fig


def fig4_box_plots(df, save=True):
    """
    Box plots: Score distributions per dataset for each metric.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = [
        ("rouge_l", "ROUGE-L F1"),
        ("bertscore_f1", "BERTScore F1"),
        ("judge_score", "Judge Score (1-5)"),
    ]
    
    for ax, (col, title) in zip(axes, metrics):
        order = df.groupby("dataset")[col].mean().sort_values(ascending=False).index
        
        sns.boxplot(
            data=df, x="dataset", y=col, order=order,
            palette="husl", ax=ax, linewidth=0.8,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    fig.suptitle("Score Distributions by Dataset", fontweight="bold", fontsize=14)
    plt.tight_layout()
    
    if save:
        path = config.FIGURES_DIR / "fig4_box_plots.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    
    plt.close(fig)
    return fig


def fig5_correlation_scatter(df, save=True):
    """
    Scatter plots: Pairwise metric correlations (ROUGE vs BERTScore vs Judge).
    """
    metrics = [
        ("rouge_l", "ROUGE-L"),
        ("bertscore_f1", "BERTScore F1"),
        ("judge_score", "Judge Score"),
    ]
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (i, j) in zip(axes, pairs):
        col_i, label_i = metrics[i]
        col_j, label_j = metrics[j]
        
        df_clean = df[[col_i, col_j, "dataset"]].dropna()
        
        sns.scatterplot(
            data=df_clean, x=col_i, y=col_j,
            hue="dataset", alpha=0.6, ax=ax, palette="husl",
            legend=(i == 0 and j == 1),  # Only show legend on first plot
        )
        
        # Add trend line
        from scipy.stats import spearmanr
        rho, p = spearmanr(df_clean[col_i], df_clean[col_j])
        z = np.polyfit(df_clean[col_i], df_clean[col_j], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(df_clean[col_i].min(), df_clean[col_i].max(), 100)
        ax.plot(x_range, p_line(x_range), "--", color="gray", alpha=0.7)
        
        ax.set_xlabel(label_i)
        ax.set_ylabel(label_j)
        ax.set_title(f"{label_i} vs {label_j}\n(ρ={rho:.3f}, p={p:.1e})", fontweight="bold")
        ax.grid(alpha=0.3)
    
    fig.suptitle("Metric Correlation Analysis", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save:
        path = config.FIGURES_DIR / "fig5_correlation_scatter.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    
    plt.close(fig)
    return fig


def fig6_latency_chart(df, save=True):
    """
    Bar chart: Average generation latency per dataset.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    latency = df.groupby("dataset")["latency_sec"].agg(["mean", "std"]).sort_values("mean")
    
    bars = ax.barh(
        range(len(latency)), latency["mean"].values,
        xerr=latency["std"].values,
        color=sns.color_palette("viridis", len(latency)),
        edgecolor="white", linewidth=0.5, capsize=3,
    )
    
    ax.set_yticks(range(len(latency)))
    ax.set_yticklabels(latency.index)
    ax.set_xlabel("Average Latency (seconds)")
    ax.set_title(f"Generation Latency by Dataset — {config.MODEL_DISPLAY_NAME}", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(latency["mean"], latency["std"])):
        ax.text(mean + std + 0.5, i, f"{mean:.1f}s", va="center", fontsize=9)
    
    plt.tight_layout()
    
    if save:
        path = config.FIGURES_DIR / "fig6_latency_chart.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    
    plt.close(fig)
    return fig


def fig7_mcq_accuracy(df, save=True):
    """
    Bar chart: MCQ accuracy per dataset (only for MCQ-type datasets).
    """
    mcq_df = df[df["mcq_correct"].notna()].copy()
    
    if mcq_df.empty:
        print("  ⚠️ No MCQ data available, skipping MCQ accuracy chart")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    accuracy = mcq_df.groupby("dataset")["mcq_correct"].mean().sort_values(ascending=False)
    counts = mcq_df.groupby("dataset")["mcq_correct"].count()
    
    colors = ["#2ecc71" if acc >= 0.5 else "#e74c3c" for acc in accuracy.values]
    
    bars = ax.barh(
        range(len(accuracy)), accuracy.values * 100,
        color=colors, edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    
    ax.set_yticks(range(len(accuracy)))
    ax.set_yticklabels([f"{ds} (n={counts[ds]})" for ds in accuracy.index])
    ax.set_xlabel("MCQ Accuracy (%)")
    ax.set_title(f"Multiple-Choice Accuracy — {config.MODEL_DISPLAY_NAME}", fontweight="bold")
    ax.axvline(x=25, color="red", linestyle="--", alpha=0.5, label="Random Baseline (25%)")
    ax.axvline(x=50, color="orange", linestyle="--", alpha=0.5, label="50% Threshold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 105)
    
    # Add percentage labels
    for i, acc in enumerate(accuracy.values):
        ax.text(acc * 100 + 1, i, f"{acc*100:.1f}%", va="center", fontsize=9)
    
    plt.tight_layout()
    
    if save:
        path = config.FIGURES_DIR / "fig7_mcq_accuracy.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    
    plt.close(fig)
    return fig


def generate_all_figures(df=None):
    """Generate all figures and save to results/figures/."""
    if df is None:
        df, leaderboard = load_data()
    
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    fig1_dataset_metric_bars(df)
    fig2_radar_chart(df)
    fig3_heatmap(df)
    fig4_box_plots(df)
    fig5_correlation_scatter(df)
    fig6_latency_chart(df)
    fig7_mcq_accuracy(df)
    
    print(f"\n✓ All figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
