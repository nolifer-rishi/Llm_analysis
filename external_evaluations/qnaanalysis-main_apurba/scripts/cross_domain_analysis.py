"""
EduBench-Local — Cross-Domain Analysis Engine (SciQ × RACE)
------------------------------------------------------------
The flagship cross-domain comparison script implementing the paper's key novelty:
contrasting model performance on Science QA (SciQ) vs Language/Reading Comprehension (RACE).

Analyses performed:
  1. Per-domain descriptive statistics (mean, std, 95% CI) for all 3 metrics
  2. Mann-Whitney U test (non-parametric, no normality assumption) between the 2 domains
  3. Cohen's d effect size estimation per metric
  4. Publication-ready figures:
     - cross_domain_bar.png       : Side-by-side metric comparison (grouped bar)
     - cross_domain_violin.png    : Score distributions (violin + strip) per domain
     - cross_domain_scatter.png   : BERTScore vs Judge scatter, colored by domain
     - cross_domain_radar.png     : Radar chart: SciQ vs RACE across 3 metrics
  5. cross_domain_summary.csv     : Full statistical report

Usage:
  python scripts/cross_domain_analysis.py [--input results/scored_results.json] [--output-dir plots]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from scipy.stats import mannwhitneyu, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[!] scipy not installed. Statistical tests will be skipped.")

# -----------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------
DOMAIN_COLORS = {
    "Science (SciQ)": "#3B82F6",         # Blue
    "Language/Reading (RACE)": "#10B981", # Emerald
}
DOMAIN_SHORT = {
    "Science (SciQ)": "SciQ\n(Science)",
    "Language/Reading (RACE)": "RACE\n(Language)",
}
METRIC_LABELS = {
    "judge_score":   "LLM-as-Judge (1–5)",
    "bertscore_f1":  "BERTScore F1",
    "rouge_l":       "ROUGE-L F1",
}
METRIC_SCALE = {
    "judge_score":  (1.0, 5.0),
    "bertscore_f1": (0.0, 1.0),
    "rouge_l":      (0.0, 1.0),
}

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor":   "#0F172A",
    "figure.facecolor": "#080D1A",
    "grid.color":       "#1E293B",
    "grid.linestyle":   "--",
    "text.color":       "#F1F5F9",
    "axes.labelcolor":  "#CBD5E1",
    "xtick.color":      "#94A3B8",
    "ytick.color":      "#94A3B8",
    "axes.edgecolor":   "#334155",
    "axes.titlecolor":  "#F8FAFC",
})
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     14,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.titlesize":   16,
    "figure.autolayout":  True,
})


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def load_focused_data(input_path: str) -> pd.DataFrame:
    """Load scored results and filter to SciQ + RACE only."""
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(input_path)

    # Map source_dataset → domain label
    def assign_domain(row):
        src = str(row.get("source_dataset", "")).strip()
        qid = str(row.get("id", "")).lower()
        if src == "SciQ" or qid.startswith("sciq"):
            return "Science (SciQ)"
        if src == "RACE" or qid.startswith("race"):
            return "Language/Reading (RACE)"
        return None  # Other datasets — will be filtered out

    df["domain"] = df.apply(assign_domain, axis=1)
    focused = df[df["domain"].notna()].copy()

    if len(focused) == 0:
        raise ValueError("No SciQ or RACE records found in the input file.")

    print(f"[*] Loaded {len(focused)} focused records "
          f"(SciQ={len(focused[focused['domain']=='Science (SciQ)'])}, "
          f"RACE={len(focused[focused['domain']=='Language/Reading (RACE)'])})")
    return focused


# -----------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------
def compute_domain_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-domain descriptive statistics with 95% CI."""
    rows = []
    domains = df["domain"].unique()
    for domain in sorted(domains):
        sub = df[df["domain"] == domain]
        for metric, label in METRIC_LABELS.items():
            if metric not in sub.columns:
                continue
            vals = sub[metric].dropna()
            n = len(vals)
            mean = vals.mean()
            std  = vals.std()
            se   = std / np.sqrt(n)
            ci95 = 1.96 * se
            rows.append({
                "domain":  domain,
                "metric":  metric,
                "label":   label,
                "n":       n,
                "mean":    round(mean, 4),
                "std":     round(std,  4),
                "se":      round(se,   4),
                "ci95":    round(ci95, 4),
                "min":     round(vals.min(), 4),
                "max":     round(vals.max(), 4),
                "median":  round(vals.median(), 4),
            })
    return pd.DataFrame(rows)


def compute_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U tests + Cohen's d effect sizes between the 2 domains."""
    domains = sorted(df["domain"].unique())
    if len(domains) < 2:
        return pd.DataFrame()

    d1_label, d2_label = domains[0], domains[1]
    d1 = df[df["domain"] == d1_label]
    d2 = df[df["domain"] == d2_label]

    rows = []
    for metric, label in METRIC_LABELS.items():
        if metric not in df.columns:
            continue
        s1 = d1[metric].dropna().values
        s2 = d2[metric].dropna().values

        # Mann-Whitney U
        p_val = np.nan
        if SCIPY_AVAILABLE and len(s1) > 2 and len(s2) > 2:
            try:
                _, p_val = mannwhitneyu(s1, s2, alternative="two-sided")
                p_val = round(p_val, 6)
            except Exception:
                pass

        # Cohen's d (pooled std)
        pooled_std = np.sqrt(((len(s1)-1)*s1.std()**2 + (len(s2)-1)*s2.std()**2) /
                             (len(s1) + len(s2) - 2))
        cohens_d = (s1.mean() - s2.mean()) / pooled_std if pooled_std > 0 else 0.0

        rows.append({
            "metric":              metric,
            "label":               label,
            f"mean_{d1_label}":    round(s1.mean(), 4),
            f"mean_{d2_label}":    round(s2.mean(), 4),
            "diff":                round(s1.mean() - s2.mean(), 4),
            "cohens_d":            round(cohens_d, 4),
            "mannwhitney_p":       p_val,
            "significant_p05":     "Yes" if isinstance(p_val, float) and p_val < 0.05 else "No",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Plot 1: Grouped bar — side-by-side metrics across 2 domains
# -----------------------------------------------------------------------
def plot_cross_domain_bar(stats_df: pd.DataFrame, sig_df: pd.DataFrame, output_dir: str):
    metrics = list(METRIC_LABELS.keys())
    domains = sorted(stats_df["domain"].unique())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 5.5))
    fig.patch.set_facecolor("#080D1A")

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("#0F172A")
        sub = stats_df[stats_df["metric"] == metric].set_index("domain")

        bars = []
        for di, domain in enumerate(domains):
            if domain not in sub.index:
                continue
            row = sub.loc[domain]
            color = DOMAIN_COLORS.get(domain, "#64748B")
            bar = ax.bar(
                di, row["mean"], 0.55,
                color=color, alpha=0.88,
                edgecolor=color, linewidth=1.5,
                label=domain
            )
            # Error bar (95% CI)
            ax.errorbar(di, row["mean"], yerr=row["ci95"],
                        fmt="none", color="#F8FAFC", capsize=5, linewidth=1.5, capthick=1.5)
            # Value annotation
            ax.text(di, row["mean"] + row["ci95"] + (METRIC_SCALE[metric][1] - METRIC_SCALE[metric][0]) * 0.03,
                    f"{row['mean']:.3f}", ha="center", va="bottom",
                    color="#F8FAFC", fontweight="bold", fontsize=10)
            bars.append(bar)

        # Significance annotation
        if not sig_df.empty and metric in sig_df["metric"].values:
            sig_row = sig_df[sig_df["metric"] == metric].iloc[0]
            p = sig_row.get("mannwhitney_p", np.nan)
            sig_str = ""
            if isinstance(p, float) and not np.isnan(p):
                if p < 0.001:
                    sig_str = "*** p<0.001"
                elif p < 0.01:
                    sig_str = f"** p={p:.3f}"
                elif p < 0.05:
                    sig_str = f"* p={p:.3f}"
                else:
                    sig_str = f"ns (p={p:.3f})"
            if sig_str:
                y_max = stats_df[stats_df["metric"] == metric]["mean"].max()
                y_top = y_max + stats_df[stats_df["metric"] == metric]["ci95"].max()
                ax.annotate(sig_str, xy=(0.5, 1.02), xycoords="axes fraction",
                            ha="center", fontsize=9, color="#FBBF24", fontweight="bold")

        ymin, ymax = METRIC_SCALE[metric]
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.22)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([DOMAIN_SHORT.get(d, d) for d in domains],
                           fontsize=10, fontweight="bold")
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold",
                     color="#F8FAFC", pad=8)
        ax.set_ylabel("Score", fontsize=10, color="#CBD5E1")

    fig.suptitle("Cross-Domain Performance Comparison: SciQ (Science) vs. RACE (Language/Reading)",
                 fontsize=14, fontweight="bold", color="#F8FAFC", y=1.02)

    legend_patches = [mpatches.Patch(color=DOMAIN_COLORS[d], label=d) for d in domains if d in DOMAIN_COLORS]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.2, edgecolor="#334155",
               labelcolor="#F1F5F9", fontsize=10)

    out_path = os.path.join(output_dir, "cross_domain_bar.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#080D1A")
    plt.close()
    print(f"[+] Saved: {out_path}")


# -----------------------------------------------------------------------
# Plot 2: Violin + strip — score distributions per domain
# -----------------------------------------------------------------------
def plot_cross_domain_violin(df: pd.DataFrame, output_dir: str):
    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5.5))
    fig.patch.set_facecolor("#080D1A")

    palette = {d: DOMAIN_COLORS.get(d, "#64748B") for d in df["domain"].unique()}

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("#0F172A")

        plot_data = df[["domain", metric]].dropna()

        # Violin
        sns.violinplot(
            data=plot_data, x="domain", y=metric,
            palette=palette, ax=ax, inner=None, alpha=0.6,
            linewidth=1.2, cut=0
        )
        # Strip (individual points)
        sns.stripplot(
            data=plot_data, x="domain", y=metric,
            palette=palette, ax=ax, size=2.5, alpha=0.35, jitter=True
        )
        # Median line
        for di, domain in enumerate(sorted(df["domain"].unique())):
            med = df[df["domain"] == domain][metric].median()
            ax.hlines(med, di - 0.3, di + 0.3, colors="#FBBF24", linewidth=2, label="Median" if di == 0 else "")

        ymin, ymax = METRIC_SCALE[metric]
        ax.set_ylim(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.1)
        ax.set_xticklabels([DOMAIN_SHORT.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                           fontsize=9, fontweight="bold")
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold",
                     color="#F8FAFC", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=10, color="#CBD5E1")

    fig.suptitle("Score Distribution: SciQ vs. RACE (Violin + Individual Points)",
                 fontsize=14, fontweight="bold", color="#F8FAFC", y=1.02)

    out_path = os.path.join(output_dir, "cross_domain_violin.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#080D1A")
    plt.close()
    print(f"[+] Saved: {out_path}")


# -----------------------------------------------------------------------
# Plot 3: Scatter — BERTScore vs LLM-Judge per domain
# -----------------------------------------------------------------------
def plot_cross_domain_scatter(df: pd.DataFrame, output_dir: str):
    if "bertscore_f1" not in df.columns or "judge_score" not in df.columns:
        print("[!] Skipping scatter — missing columns.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#080D1A")
    ax.set_facecolor("#0F172A")

    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain].dropna(subset=["bertscore_f1", "judge_score"])
        color = DOMAIN_COLORS.get(domain, "#64748B")
        ax.scatter(sub["bertscore_f1"], sub["judge_score"],
                   color=color, alpha=0.45, s=30, label=domain, edgecolors="none")

        # Domain centroid + CI ellipse placeholder (mean ± std)
        mx, my = sub["bertscore_f1"].mean(), sub["judge_score"].mean()
        sx, sy = sub["bertscore_f1"].std(), sub["judge_score"].std()
        ax.scatter(mx, my, color=color, s=200, marker="D", edgecolors="#FFFFFF",
                   linewidth=1.5, zorder=10)
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="none",
                    color=color, alpha=0.7, capsize=5, linewidth=1.5)
        ax.text(mx + 0.003, my + 0.08, f"  {domain.split('(')[1].rstrip(')')}\nμ=({mx:.3f}, {my:.2f})",
                fontsize=9, color="#F8FAFC", fontweight="bold")

    ax.set_xlabel("BERTScore F1 (Semantic Similarity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("LLM-as-Judge Score (1–5, Pedagogical Correctness)", fontsize=12, fontweight="bold")
    ax.set_title("BERTScore vs. LLM-Judge: SciQ (Science) vs. RACE (Language/Reading)",
                 fontsize=13, fontweight="bold", color="#F8FAFC", pad=10)

    ax.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=10)

    # Correlation note
    corr = df[["bertscore_f1", "judge_score"]].dropna().corr().iloc[0, 1]
    ax.text(0.02, 0.98, f"Overall r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, va="top", color="#94A3B8", style="italic")

    out_path = os.path.join(output_dir, "cross_domain_scatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#080D1A")
    plt.close()
    print(f"[+] Saved: {out_path}")


# -----------------------------------------------------------------------
# Plot 4: Radar — SciQ vs RACE across 3 normalized metrics
# -----------------------------------------------------------------------
def plot_cross_domain_radar(stats_df: pd.DataFrame, output_dir: str):
    categories = ["LLM-Judge\n(Normalized)", "BERTScore\nF1", "ROUGE-L\nF1"]
    metric_keys = ["judge_score", "bertscore_f1", "rouge_l"]
    # Normalize judge to [0,1] range for fair radar comparison
    normalizers = {"judge_score": 4.0, "bertscore_f1": 1.0, "rouge_l": 1.0}
    offsets = {"judge_score": 1.0, "bertscore_f1": 0.0, "rouge_l": 0.0}

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#080D1A")
    ax.set_facecolor("#0F172A")

    domains = sorted(stats_df["domain"].unique())

    for domain in domains:
        sub = stats_df[stats_df["domain"] == domain].set_index("metric")
        values = []
        for m in metric_keys:
            if m in sub.index:
                raw = sub.loc[m, "mean"]
                norm_val = (raw - offsets[m]) / normalizers[m]
                values.append(round(max(0, min(1, norm_val)), 4))
            else:
                values.append(0.0)
        values += values[:1]

        color = DOMAIN_COLORS.get(domain, "#64748B")
        ax.plot(angles, values, "o-", linewidth=2.5, color=color, label=domain)
        ax.fill(angles, values, alpha=0.18, color=color)

        # Label each vertex
        for angle, val, cat in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, val + 0.06, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11, color="#F1F5F9")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#94A3B8")
    ax.grid(color="#1E293B", linewidth=0.8)
    ax.spines["polar"].set_color("#334155")

    ax.set_title("Cross-Domain Tri-Metric Radar: SciQ vs. RACE\n(All metrics normalized to [0, 1])",
                 size=13, fontweight="bold", color="#F8FAFC", y=1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#0F172A", edgecolor="#334155", labelcolor="#F1F5F9", fontsize=10)

    out_path = os.path.join(output_dir, "cross_domain_radar.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#080D1A")
    plt.close()
    print(f"[+] Saved: {out_path}")


# -----------------------------------------------------------------------
# Console report
# -----------------------------------------------------------------------
def print_report(stats_df: pd.DataFrame, sig_df: pd.DataFrame):
    print("\n" + "="*90)
    print("CROSS-DOMAIN ANALYSIS REPORT: SciQ (Science) × RACE (Language/Reading)")
    print("="*90)

    print("\n--- DESCRIPTIVE STATISTICS ---")
    pivot = stats_df.pivot_table(index="label", columns="domain",
                                  values=["mean", "std", "ci95"])
    print(pivot.to_string())

    if not sig_df.empty:
        print("\n--- STATISTICAL SIGNIFICANCE (Mann-Whitney U Test) ---")
        display_cols = [c for c in sig_df.columns if c != "metric"]
        print(sig_df[display_cols].to_string(index=False))

    print("\n" + "="*90)
    print("KEY FINDINGS:")
    for _, row in sig_df.iterrows():
        p = row.get("mannwhitney_p", np.nan)
        d = row.get("cohens_d", 0)
        sig = row.get("significant_p05", "No")
        magnitude = "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
        print(f"  {row['label']}: "
              f"diff={row['diff']:+.4f}, Cohen's d={d:.3f} ({magnitude} effect), "
              f"p={p:.4f} [{'SIGNIFICANT' if sig == 'Yes' else 'not significant'}]")
    print("="*90 + "\n")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Analysis Engine: SciQ x RACE")
    parser.add_argument("--input", type=str, default="results/scored_results.json",
                        help="Path to scored results JSON or CSV")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to save plots and summary CSV")
    args = parser.parse_args()

    # Resolve input path
    input_path = args.input
    if not os.path.exists(input_path):
        for alt in ["results/scored_results.json", "results/scored_results.csv",
                    "scored_results.json", "scored_results.csv"]:
            if os.path.exists(alt):
                input_path = alt
                break
        else:
            print(f"[!] Error: Input file not found. Run generate_answers.py + evaluate_metrics.py first.")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and filter to SciQ + RACE only
    df = load_focused_data(input_path)

    # Statistics
    stats_df = compute_domain_stats(df)
    sig_df   = compute_significance(df)

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, "cross_domain_summary.csv")
    stats_df.to_csv(summary_path, index=False)
    print(f"[+] Saved domain statistics: {summary_path}")

    if not sig_df.empty:
        sig_path = os.path.join(args.output_dir, "cross_domain_significance.csv")
        sig_df.to_csv(sig_path, index=False)
        print(f"[+] Saved significance tests: {sig_path}")

    # Console report
    print_report(stats_df, sig_df)

    # Plots
    print("[*] Generating cross-domain publication figures...")
    plot_cross_domain_bar(stats_df, sig_df, args.output_dir)
    plot_cross_domain_violin(df, args.output_dir)
    plot_cross_domain_scatter(df, args.output_dir)
    plot_cross_domain_radar(stats_df, args.output_dir)

    print(f"\n[+] Cross-domain analysis complete! "
          f"Figures and summaries saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
