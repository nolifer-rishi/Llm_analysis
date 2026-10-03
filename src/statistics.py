"""
EduBench-Local: Statistical Analysis
======================================
Cross-dataset significance tests, metric correlation analysis,
and distribution comparisons.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from scipy.stats import (
    wilcoxon,
    kruskal,
    spearmanr,
    mannwhitneyu,
    shapiro,
)


def load_scored_results():
    """Load scored results from CSV."""
    path = config.SCORED_RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Scored results not found at {path}")
    return pd.read_csv(path)


def pairwise_wilcoxon_tests(df, metric="judge_score"):
    """
    Run pairwise Wilcoxon signed-rank tests between all dataset pairs
    for a given metric.
    
    Note: Wilcoxon signed-rank requires paired samples of equal length.
    Since our datasets have the same sample size (100), we pair by position index.
    If sizes differ, we use Mann-Whitney U instead.
    
    Returns:
        pd.DataFrame with columns: dataset_a, dataset_b, statistic, p_value, significant
    """
    datasets = df["dataset"].unique()
    results = []
    
    for ds_a, ds_b in combinations(datasets, 2):
        scores_a = df[df["dataset"] == ds_a][metric].dropna().values
        scores_b = df[df["dataset"] == ds_b][metric].dropna().values
        
        try:
            if len(scores_a) == len(scores_b):
                stat, p_val = wilcoxon(scores_a, scores_b)
                test_type = "Wilcoxon"
            else:
                stat, p_val = mannwhitneyu(scores_a, scores_b, alternative="two-sided")
                test_type = "Mann-Whitney U"
            
            results.append({
                "dataset_a": ds_a,
                "dataset_b": ds_b,
                "test": test_type,
                "metric": metric,
                "statistic": round(stat, 4),
                "p_value": round(p_val, 6),
                "significant_0.05": p_val < 0.05,
                "significant_0.01": p_val < 0.01,
            })
        except Exception as e:
            results.append({
                "dataset_a": ds_a,
                "dataset_b": ds_b,
                "test": "ERROR",
                "metric": metric,
                "statistic": None,
                "p_value": None,
                "significant_0.05": None,
                "significant_0.01": None,
            })
    
    return pd.DataFrame(results)


def kruskal_wallis_test(df, metric="judge_score"):
    """
    Run Kruskal-Wallis H-test across all datasets for a given metric.
    Tests whether score distributions differ significantly across datasets.
    
    Returns:
        dict with statistic, p_value, significant
    """
    groups = [
        group[metric].dropna().values
        for _, group in df.groupby("dataset")
    ]
    
    # Need at least 2 groups with data
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {"statistic": None, "p_value": None, "significant": None}
    
    stat, p_val = kruskal(*groups)
    
    return {
        "test": "Kruskal-Wallis H",
        "metric": metric,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "significant_0.05": p_val < 0.05,
    }


def metric_correlation_analysis(df):
    """
    Compute Spearman's rank correlation between the 3 metrics.
    Tests whether ROUGE-L, BERTScore, and Judge scores agree.
    
    Returns:
        pd.DataFrame correlation matrix with p-values
    """
    metrics = ["rouge_l", "bertscore_f1", "judge_score"]
    
    # Drop rows with any NaN in the metrics
    df_clean = df[metrics].dropna()
    
    results = []
    
    for m1, m2 in combinations(metrics, 2):
        rho, p_val = spearmanr(df_clean[m1], df_clean[m2])
        results.append({
            "metric_a": m1,
            "metric_b": m2,
            "spearman_rho": round(rho, 4),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
            "correlation_strength": _interpret_correlation(abs(rho)),
        })
    
    return pd.DataFrame(results)


def _interpret_correlation(rho_abs):
    """Interpret Spearman's rho magnitude."""
    if rho_abs >= 0.8:
        return "Very Strong"
    elif rho_abs >= 0.6:
        return "Strong"
    elif rho_abs >= 0.4:
        return "Moderate"
    elif rho_abs >= 0.2:
        return "Weak"
    else:
        return "Very Weak / None"


def distribution_analysis(df):
    """
    Analyze score distributions per dataset: normality test, skewness, kurtosis.
    """
    results = []
    
    for dataset in df["dataset"].unique():
        ds_data = df[df["dataset"] == dataset]
        
        for metric in ["rouge_l", "bertscore_f1", "judge_score"]:
            values = ds_data[metric].dropna().values
            
            if len(values) < 8:
                continue
            
            # Shapiro-Wilk normality test
            try:
                sw_stat, sw_p = shapiro(values)
            except Exception:
                sw_stat, sw_p = None, None
            
            results.append({
                "dataset": dataset,
                "metric": metric,
                "n": len(values),
                "mean": round(float(np.mean(values)), 4),
                "median": round(float(np.median(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "skewness": round(float(pd.Series(values).skew()), 4),
                "kurtosis": round(float(pd.Series(values).kurtosis()), 4),
                "shapiro_w": round(sw_stat, 4) if sw_stat else None,
                "shapiro_p": round(sw_p, 6) if sw_p else None,
                "is_normal_0.05": sw_p > 0.05 if sw_p else None,
            })
    
    return pd.DataFrame(results)


def run_all_statistics(df=None):
    """
    Run all statistical analyses and return results.
    
    Returns:
        dict with all statistical analysis results
    """
    if df is None:
        df = load_scored_results()
    
    print("\n" + "=" * 60)
    print("Statistical Analysis")
    print("=" * 60)
    
    results = {}
    
    # 1. Kruskal-Wallis tests
    print("\n[1] Kruskal-Wallis H-test (do datasets differ?)")
    for metric in ["rouge_l", "bertscore_f1", "judge_score"]:
        kw = kruskal_wallis_test(df, metric)
        print(f"  {metric}: H={kw['statistic']}, p={kw['p_value']}, sig={kw['significant_0.05']}")
        results[f"kruskal_{metric}"] = kw
    
    # 2. Pairwise Wilcoxon tests
    print("\n[2] Pairwise significance tests")
    for metric in ["rouge_l", "bertscore_f1", "judge_score"]:
        pw = pairwise_wilcoxon_tests(df, metric)
        sig_count = pw["significant_0.05"].sum()
        total = len(pw)
        print(f"  {metric}: {sig_count}/{total} pairs significantly different (p<0.05)")
        results[f"pairwise_{metric}"] = pw
    
    # 3. Metric correlation
    print("\n[3] Metric correlation (Spearman's ρ)")
    corr = metric_correlation_analysis(df)
    for _, row in corr.iterrows():
        print(f"  {row['metric_a']} vs {row['metric_b']}: "
              f"ρ={row['spearman_rho']}, p={row['p_value']} ({row['correlation_strength']})")
    results["metric_correlations"] = corr
    
    # 4. Distribution analysis
    print("\n[4] Distribution analysis")
    dist = distribution_analysis(df)
    results["distributions"] = dist
    
    # Save all results
    output_dir = config.RESULTS_DIR / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corr.to_csv(output_dir / "metric_correlations.csv", index=False)
    dist.to_csv(output_dir / "distributions.csv", index=False)
    
    for metric in ["rouge_l", "bertscore_f1", "judge_score"]:
        pw = results[f"pairwise_{metric}"]
        pw.to_csv(output_dir / f"pairwise_{metric}.csv", index=False)
    
    print(f"\n✓ Statistical results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    stats = run_all_statistics()
