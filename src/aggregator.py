"""
EduBench-Local: Aggregator
============================
Aggregates scored results into per-dataset rankings, normalized scores,
and combined leaderboard.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_scored_results():
    """Load scored results from CSV."""
    path = config.SCORED_RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Scored results not found at {path}. "
            "Run the evaluation step first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} scored results")
    return df


def compute_dataset_summary(df):
    """
    Compute summary statistics per dataset.
    
    Returns:
        pd.DataFrame with per-dataset averages, std devs, and counts
    """
    summary = df.groupby("dataset").agg(
        count=("id", "count"),
        avg_rouge_l=("rouge_l", "mean"),
        std_rouge_l=("rouge_l", "std"),
        avg_bertscore=("bertscore_f1", "mean"),
        std_bertscore=("bertscore_f1", "std"),
        avg_judge=("judge_score", "mean"),
        std_judge=("judge_score", "std"),
        avg_latency=("latency_sec", "mean"),
        avg_tokens=("token_count", "mean"),
    ).reset_index()
    
    # Add MCQ accuracy where applicable
    mcq_df = df[df["mcq_correct"].notna()]
    if not mcq_df.empty:
        mcq_acc = mcq_df.groupby("dataset").agg(
            mcq_accuracy=("mcq_correct", "mean"),
            mcq_count=("mcq_correct", "count"),
        ).reset_index()
        summary = summary.merge(mcq_acc, on="dataset", how="left")
    else:
        summary["mcq_accuracy"] = None
        summary["mcq_count"] = 0
    
    return summary


def compute_overall_summary(df):
    """Compute overall (across all datasets) summary statistics."""
    overall = {
        "total_questions": len(df),
        "avg_rouge_l": df["rouge_l"].mean(),
        "std_rouge_l": df["rouge_l"].std(),
        "avg_bertscore": df["bertscore_f1"].mean(),
        "std_bertscore": df["bertscore_f1"].std(),
        "avg_judge": df["judge_score"].mean(),
        "std_judge": df["judge_score"].std(),
        "avg_latency": df["latency_sec"].mean(),
        "total_latency_min": df["latency_sec"].sum() / 60,
        "avg_tokens": df["token_count"].mean(),
    }
    
    # MCQ accuracy overall
    mcq_df = df[df["mcq_correct"].notna()]
    if not mcq_df.empty:
        overall["mcq_accuracy"] = mcq_df["mcq_correct"].mean()
        overall["mcq_count"] = len(mcq_df)
    
    return overall


def compute_normalized_scores(dataset_summary):
    """
    Normalize each metric to 0-1 range across datasets for fair comparison.
    """
    summary = dataset_summary.copy()
    
    for col in ["avg_rouge_l", "avg_bertscore", "avg_judge"]:
        col_min = summary[col].min()
        col_max = summary[col].max()
        if col_max > col_min:
            summary[f"{col}_norm"] = (summary[col] - col_min) / (col_max - col_min)
        else:
            summary[f"{col}_norm"] = 1.0
    
    # Combined score (mean of normalized metrics)
    norm_cols = [c for c in summary.columns if c.endswith("_norm")]
    summary["combined_score"] = summary[norm_cols].mean(axis=1)
    
    # Rank datasets by combined score
    summary["rank"] = summary["combined_score"].rank(ascending=False).astype(int)
    summary = summary.sort_values("rank")
    
    return summary


def compute_question_type_summary(df):
    """Compare performance on open-ended vs MCQ questions."""
    type_summary = df.groupby("question_type").agg(
        count=("id", "count"),
        avg_rouge_l=("rouge_l", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
        avg_judge=("judge_score", "mean"),
        avg_latency=("latency_sec", "mean"),
    ).reset_index()
    
    return type_summary


def build_leaderboard(df=None):
    """
    Build the full leaderboard with all aggregations.
    
    Returns:
        dict with keys: dataset_summary, overall_summary, normalized, question_type_summary
    """
    if df is None:
        df = load_scored_results()
    
    print("\n" + "=" * 60)
    print("Building Leaderboard")
    print("=" * 60)
    
    # Per-dataset summary
    dataset_summary = compute_dataset_summary(df)
    print("\n[Per-Dataset Summary]")
    print(dataset_summary[["dataset", "count", "avg_rouge_l", "avg_bertscore", "avg_judge"]].to_string(index=False))
    
    # Overall summary
    overall = compute_overall_summary(df)
    print(f"\n[Overall Summary]")
    print(f"  ROUGE-L:   {overall['avg_rouge_l']:.4f} ± {overall['std_rouge_l']:.4f}")
    print(f"  BERTScore: {overall['avg_bertscore']:.4f} ± {overall['std_bertscore']:.4f}")
    print(f"  Judge:     {overall['avg_judge']:.2f} ± {overall['std_judge']:.2f}")
    print(f"  Latency:   {overall['avg_latency']:.1f}s avg ({overall['total_latency_min']:.1f} min total)")
    
    # Normalized + ranked
    normalized = compute_normalized_scores(dataset_summary)
    print(f"\n[Dataset Rankings (by Combined Score)]")
    print(normalized[["rank", "dataset", "combined_score", "avg_rouge_l_norm", "avg_bertscore_norm", "avg_judge_norm"]].to_string(index=False))
    
    # Question type comparison
    qt_summary = compute_question_type_summary(df)
    print(f"\n[Question Type Comparison]")
    print(qt_summary.to_string(index=False))
    
    # Save leaderboard
    normalized.to_csv(config.LEADERBOARD_FILE, index=False)
    print(f"\n✓ Leaderboard saved to {config.LEADERBOARD_FILE}")
    
    return {
        "dataset_summary": dataset_summary,
        "overall_summary": overall,
        "normalized": normalized,
        "question_type_summary": qt_summary,
    }


if __name__ == "__main__":
    leaderboard = build_leaderboard()
