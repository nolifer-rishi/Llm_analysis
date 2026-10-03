"""
EduBench-Local — Ranking, Statistical Analysis & Visualization Engine
----------------------------------------------------------------------
Processes scored results to:
  1. Compute normalized composite leaderboard rankings (ROUGE-L, BERTScore, LLM-Judge)
  2. Compute subject-wise & dataset-wise performance breakdown (SciQ, ARC, OpenBookQA, RACE, SQuAD)
  3. Perform pairwise Wilcoxon signed-rank statistical significance tests
  4. Generate publication-ready high-resolution figures:
     - leaderboard_bar.png
     - subject_heatmap.png (Rich Dataset x Metric and Model x Dataset Heatmaps)
     - dataset_breakdown.png
     - radar_metrics.png
     - efficiency_frontier.png

Usage:
  python analyze_and_plot.py [--input results/scored_results.csv] [--output-dir plots]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

# Visual styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.autolayout": True
})


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. Returns 1.0 if all values are identical."""
    if series.max() == series.min():
        return pd.Series(1.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Computes overall model leaderboard with normalized combined score."""
    leaderboard = df.groupby("model").agg(
        total_questions=("id", "count"),
        avg_rouge=("rouge_l", "mean"),
        std_rouge=("rouge_l", "std"),
        avg_bertscore=("bertscore_f1", "mean"),
        std_bertscore=("bertscore_f1", "std"),
        avg_judge=("judge_score", "mean"),
        std_judge=("judge_score", "std"),
        avg_latency=("latency_sec", "mean")
    ).reset_index()

    leaderboard = leaderboard.fillna(0.0)

    # Normalize metrics to [0, 1]
    leaderboard["rouge_norm"] = min_max_normalize(leaderboard["avg_rouge"])
    leaderboard["bertscore_norm"] = min_max_normalize(leaderboard["avg_bertscore"])
    leaderboard["judge_norm"] = min_max_normalize(leaderboard["avg_judge"])

    # Composite combined score
    leaderboard["combined_score"] = leaderboard[
        ["rouge_norm", "bertscore_norm", "judge_norm"]
    ].mean(axis=1)

    leaderboard = leaderboard.sort_values("combined_score", ascending=False).reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1
    return leaderboard


def compute_subject_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Computes subject-specific rankings and averages."""
    subject_df = df.groupby(["subject", "model"]).agg(
        avg_rouge=("rouge_l", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
        avg_judge=("judge_score", "mean"),
        avg_latency=("latency_sec", "mean"),
        count=("id", "count")
    ).reset_index()
    return subject_df


def compute_dataset_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Computes dataset-specific breakdown (SciQ, ARC, OpenBookQA, RACE, SQuAD)."""
    if "source_dataset" not in df.columns:
        def get_source(row):
            qid = str(row["id"]).lower()
            if qid.startswith("sciq"): return "SciQ"
            if qid.startswith("arc"): return "ARC-Challenge"
            if qid.startswith("obqa"): return "OpenBookQA"
            if qid.startswith("race"): return "RACE"
            if qid.startswith("squad"): return "SQuAD"
            if qid.startswith("sci"): return "SciQ"
            if qid.startswith("rc"): return "SQuAD"
            return "General"
        df["source_dataset"] = df.apply(get_source, axis=1)

    dataset_df = df.groupby(["source_dataset", "model"]).agg(
        avg_judge=("judge_score", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
        avg_rouge=("rouge_l", "mean"),
        avg_latency=("latency_sec", "mean"),
        count=("id", "count")
    ).reset_index()
    return dataset_df


def compute_significance_tests(df: pd.DataFrame, metric: str = "judge_score") -> pd.DataFrame:
    """Computes pairwise Wilcoxon signed-rank test p-values between all models."""
    models = sorted(df["model"].unique())
    n = len(models)
    matrix = pd.DataFrame(np.ones((n, n)), index=models, columns=models)

    if wilcoxon is None:
        return matrix

    pivot = df.pivot_table(index="id", columns="model", values=metric)

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i != j and m1 in pivot.columns and m2 in pivot.columns:
                s1 = pivot[m1].dropna()
                s2 = pivot[m2].dropna()
                common_idx = s1.index.intersection(s2.index)
                if len(common_idx) > 2:
                    diff = s1.loc[common_idx] - s2.loc[common_idx]
                    if np.all(diff == 0):
                        matrix.loc[m1, m2] = 1.0
                    else:
                        try:
                            _, p_val = wilcoxon(s1.loc[common_idx], s2.loc[common_idx])
                            matrix.loc[m1, m2] = round(p_val, 4)
                        except Exception:
                            matrix.loc[m1, m2] = np.nan
    return matrix


def plot_leaderboard_bar(leaderboard: pd.DataFrame, output_dir: str):
    """Bar chart of overall model ranking by combined score."""
    plt.figure(figsize=(9, 5))
    palette = sns.color_palette("viridis", len(leaderboard))
    ax = sns.barplot(
        data=leaderboard,
        x="combined_score",
        y="model",
        hue="model",
        palette=palette,
        legend=False
    )
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f"{width:.3f}", (width + 0.01, p.get_y() + p.get_height() / 2),
                        ha="left", va="center", fontsize=10, fontweight="bold")

    plt.title("Overall LLM Leaderboard (Combined Multi-Metric Score)", pad=15, fontweight="bold")
    plt.xlabel("Combined Score (Normalized 0.0 - 1.0)")
    plt.ylabel("Model")
    plt.xlim(0, max(1.15, leaderboard["combined_score"].max() * 1.15))
    out_path = os.path.join(output_dir, "leaderboard_bar.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_subject_heatmap(dataset_df: pd.DataFrame, output_dir: str):
    """
    Rich 2D Matrix Heatmap:
    Shows performance metrics across all 5 benchmark datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    display_df = dataset_df.copy().set_index("source_dataset")
    metric_cols = {
        "avg_judge": "LLM Judge (1-5)",
        "avg_bertscore": "BERTScore (F1)",
        "avg_rouge": "ROUGE-L (F1)",
        "avg_latency": "Latency (sec)"
    }
    
    # Subplot 1: Actual Values
    table_vals = display_df[list(metric_cols.keys())].rename(columns=metric_cols)
    sns.heatmap(
        table_vals,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Metric Value"},
        ax=axes[0],
        linewidths=1,
        linecolor="#E5E7EB"
    )
    axes[0].set_title("Dataset Performance Matrix (Raw Metrics)", pad=12, fontweight="bold")
    axes[0].set_xlabel("Evaluation Metric")
    axes[0].set_ylabel("Educational Dataset Source")

    # Subplot 2: Min-Max Normalized Heatmap
    norm_vals = (table_vals - table_vals.min()) / (table_vals.max() - table_vals.min())
    if "Latency (sec)" in norm_vals.columns:
        norm_vals["Latency (sec)"] = 1.0 - norm_vals["Latency (sec)"]
        norm_vals = norm_vals.rename(columns={"Latency (sec)": "Speed Score (Inv Latency)"})

    sns.heatmap(
        norm_vals,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Normalized Score (0-1)"},
        ax=axes[1],
        linewidths=1,
        linecolor="#E5E7EB",
        vmin=0.0,
        vmax=1.0
    )
    axes[1].set_title("Normalized Cross-Domain Heatmap (0.0 - 1.0)", pad=12, fontweight="bold")
    axes[1].set_xlabel("Metric Dimension")
    axes[1].set_ylabel("")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "subject_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_dataset_breakdown(dataset_df: pd.DataFrame, output_dir: str):
    """Bar chart comparing quality (LLM-Judge 1-5) across all 5 benchmark datasets."""
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("Blues_r", len(dataset_df))
    ax = sns.barplot(
        data=dataset_df,
        x="source_dataset",
        y="avg_judge",
        hue="source_dataset",
        palette=palette,
        legend=False
    )
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}/5", (p.get_x() + p.get_width() / 2., height + 0.08),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("Educational QA Quality Across All 5 Benchmark Datasets (LLM Judge 1-5)", pad=15, fontweight="bold")
    plt.xlabel("Dataset Benchmark Source")
    plt.ylabel("Average LLM-Judge Score (1-5)")
    plt.ylim(0, 5.2)
    out_path = os.path.join(output_dir, "dataset_breakdown.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_radar_chart(leaderboard: pd.DataFrame, output_dir: str):
    """Radar chart comparing the 3 evaluation metrics per model."""
    categories = ["ROUGE-L", "BERTScore F1", "LLM-Judge (Scaled)"]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for _, row in leaderboard.iterrows():
        judge_scaled = (row["avg_judge"] - 1.0) / 4.0 if row["avg_judge"] > 0 else 0
        values = [row["avg_rouge"], row["avg_bertscore"], judge_scaled]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1.0)
    plt.title("Tri-Metric Performance Profile", size=14, y=1.08, fontweight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    out_path = os.path.join(output_dir, "radar_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_efficiency_frontier(leaderboard: pd.DataFrame, output_dir: str):
    """Scatter plot of Accuracy (Judge Score) vs. Generation Latency."""
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=leaderboard,
        x="avg_latency",
        y="avg_judge",
        hue="model",
        s=200,
        style="model",
        legend="full"
    )

    for _, row in leaderboard.iterrows():
        plt.text(
            row["avg_latency"] + 0.05,
            row["avg_judge"] + 0.03,
            f"{row['model']}\n({row['avg_judge']:.2f} pts, {row['avg_latency']:.2f}s)",
            fontsize=9
        )

    plt.title("Efficiency Frontier: Quality vs. Latency Trade-off", pad=15, fontweight="bold")
    plt.xlabel("Average Latency per Answer (seconds)")
    plt.ylabel("Average LLM-Judge Score (1-5)")
    plt.ylim(1.0, 5.2)
    out_path = os.path.join(output_dir, "efficiency_frontier.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ranking, Statistical Analysis and Visualizations for EduBench-Local")
    parser.add_argument("--input", type=str, default="scored_results.csv", help="Path to scored results CSV")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save generated plots and reports")
    args = parser.parse_args()

    # Check fallback paths
    input_path = args.input
    if not os.path.exists(input_path):
        alt = os.path.join("results", "scored_results.csv")
        if os.path.exists(alt):
            input_path = alt
        else:
            print(f"[!] Error: Input file '{args.input}' not found.")
            return

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    print(f"[*] Loaded {len(df)} scored entries from '{input_path}'.")

    # 1. Overall Leaderboard
    leaderboard = compute_leaderboard(df)
    leaderboard_csv = os.path.join(args.output_dir, "leaderboard_summary.csv")
    leaderboard.to_csv(leaderboard_csv, index=False)
    
    print("\n" + "="*95)
    print("EDUBENCH-LOCAL FINAL LEADERBOARD (750 QUESTIONS)")
    print("="*95)
    cols = ["rank", "model", "combined_score", "avg_rouge", "avg_bertscore", "avg_judge", "avg_latency"]
    print(leaderboard[cols].to_string(index=False))
    print("="*95 + "\n")

    # 2. Subject Breakdown
    subject_df = compute_subject_breakdown(df)
    subject_csv = os.path.join(args.output_dir, "subject_summary.csv")
    subject_df.to_csv(subject_csv, index=False)

    # 3. Dataset Breakdown
    dataset_df = compute_dataset_breakdown(df)
    dataset_csv = os.path.join(args.output_dir, "dataset_summary.csv")
    dataset_df.to_csv(dataset_csv, index=False)
    print("[*] 5-Dataset Benchmark Source Breakdown:")
    print(dataset_df[["source_dataset", "count", "avg_judge", "avg_bertscore", "avg_rouge", "avg_latency"]].to_string(index=False))

    # 4. Significance Testing
    if len(df["model"].unique()) > 1:
        p_matrix = compute_significance_tests(df, metric="judge_score")
        p_matrix_csv = os.path.join(args.output_dir, "wilcoxon_p_values.csv")
        p_matrix.to_csv(p_matrix_csv)

    # 5. Visualizations
    print("\n[*] Generating publication-ready figures...")
    plot_leaderboard_bar(leaderboard, args.output_dir)
    plot_subject_heatmap(dataset_df, args.output_dir)
    plot_dataset_breakdown(dataset_df, args.output_dir)
    plot_radar_chart(leaderboard, args.output_dir)
    plot_efficiency_frontier(leaderboard, args.output_dir)

    print(f"\n[+] Analysis complete! Summary files and figures stored in '{args.output_dir}/'")


if __name__ == "__main__":
    main()
