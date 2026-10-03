"""
EduBench-Local — Human Validation & Inter-Rater Agreement Tool
--------------------------------------------------------------
Validates the Local LLM-as-Judge against human ratings.
Samples N random answers, prompts the human evaluator to grade them (1-5),
and calculates:
  1. Cohen's Kappa (inter-rater agreement for categorical rating)
  2. Pearson Correlation (linear agreement)
  3. Mean Absolute Error (MAE) between Human and LLM-Judge

Usage:
  python validate_judge.py [--input scored_results.csv] [--sample-size 20] [--output-dir plots]
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr


def run_human_validation(csv_path: str, sample_size: int = 20, output_dir: str = "plots"):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("[!] Scored results are empty.")
        return

    sample_size = min(sample_size, len(df))
    sampled_indices = random.sample(range(len(df)), sample_size)
    sampled_df = df.iloc[sampled_indices].copy()

    print("\n" + "="*80)
    print(f"HUMAN VALIDATION OF LLM-AS-JUDGE ({sample_size} Question Samples)")
    print("="*80)
    print("Rating Scale:")
    print("  1 = Completely incorrect / irrelevant")
    print("  2 = Mostly incorrect / major factual errors")
    print("  3 = Partially correct / missing key points")
    print("  4 = Mostly correct and accurate with minor omissions")
    print("  5 = Fully correct, comprehensive, and accurate\n")

    human_scores = []
    llm_scores = []

    for idx, (_, row) in enumerate(sampled_df.iterrows()):
        print("-" * 80)
        print(f"Sample [{idx + 1}/{sample_size}] | Source: {row.get('source_dataset', 'General')}")
        print(f"Question: {row['question']}")
        if pd.notna(row.get('context')) and str(row.get('context')).strip():
            print(f"Context:  {str(row['context'])[:180]}...")
        print(f"Ground Truth Reference: {row['reference_answer']}")
        print(f"Model Answer:          {row['generated_answer']}")
        print("-" * 80)

        while True:
            val = input(f"Enter your human score (1-5) for Sample [{idx+1}/{sample_size}] [or 'q' to quit]: ").strip()
            if val.lower() == 'q':
                print("[*] Aborted human scoring early.")
                break
            if val in ["1", "2", "3", "4", "5"]:
                human_score = int(val)
                human_scores.append(human_score)
                llm_scores.append(int(row["judge_score"]))
                break
            else:
                print("Invalid input! Please enter an integer from 1 to 5.")

        if val.lower() == 'q':
            break

    if len(human_scores) < 3:
        print("[!] Too few samples graded to compute meaningful agreement statistics.")
        return

    # Calculate statistics
    human_arr = np.array(human_scores)
    llm_arr = np.array(llm_scores)

    # Cohen's Kappa
    kappa = cohen_kappa_score(human_arr, llm_arr)
    # Pearson r
    r_val, p_val = pearsonr(human_arr, llm_arr)
    # MAE
    mae = np.mean(np.abs(human_arr - llm_arr))
    # Exact Match %
    exact_match = np.mean(human_arr == llm_arr) * 100

    print("\n" + "="*80)
    print("INTER-RATER AGREEMENT RESULTS (Human vs. LLM-Judge)")
    print("="*80)
    print(f"Evaluated Samples:              {len(human_scores)}")
    print(f"Human Average Score:            {human_arr.mean():.2f} / 5.0")
    print(f"LLM-Judge Average Score:        {llm_arr.mean():.2f} / 5.0")
    print(f"Mean Absolute Error (MAE):      {mae:.2f} points")
    print(f"Exact Rating Agreement:         {exact_match:.1f}%")
    print(f"Pearson Correlation (r):        {r_val:.3f} (p-value: {p_val:.4f})")
    print(f"Cohen's Kappa Score (κ):        {kappa:.3f}")
    
    if kappa > 0.6:
        print("Interpretation: Substantial Agreement between Human and LLM-Judge.")
    elif kappa > 0.4:
        print("Interpretation: Moderate Agreement between Human and LLM-Judge.")
    elif kappa > 0.2:
        print("Interpretation: Fair Agreement.")
    else:
        print("Interpretation: Slight / Low Agreement.")
    print("="*80)

    # Save validation records
    val_records = pd.DataFrame({
        "sample_index": range(1, len(human_scores) + 1),
        "human_score": human_scores,
        "llm_judge_score": llm_scores,
        "abs_difference": np.abs(human_arr - llm_arr)
    })
    os.makedirs(output_dir, exist_ok=True)
    val_csv = os.path.join(output_dir, "human_validation_agreement.csv")
    val_records.to_csv(val_csv, index=False)
    print(f"[+] Human validation records saved to: {val_csv}\n")


def main():
    parser = argparse.ArgumentParser(description="Human Validation & Inter-Rater Agreement for LLM-as-Judge")
    parser.add_argument("--input", type=str, default="scored_results.csv", help="Path to scored results CSV")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of random answers to grade (default: 20)")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save agreement results")
    args = parser.parse_args()

    run_human_validation(args.input, args.sample_size, args.output_dir)


if __name__ == "__main__":
    main()
