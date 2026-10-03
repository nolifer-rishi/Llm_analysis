# -*- coding: utf-8 -*-
"""
gemini_evaluate_metrics.py
--------------------------
Evaluates generated Gemini answers from results/gemini_raw_answers.json against
reference ground-truth answers across all 5 benchmark datasets.

Datasets:
  - SciQ              (subject: science)
  - OpenBookQA        (subject: general_science)
  - ARC-Challenge     (subject: science_challenge)
  - RACE              (subject: reading_comprehension)
  - SQuAD v1.1        (subject: reading_comprehension_squad)

Metrics Computed:
  1. Exact Match (EM):
       Binary 0 or 1 after standard normalization (lowercase, punctuation, whitespace).
  2. Token F1 / Precision / Recall:
       Harmonic mean of token overlap (the gold standard for QA evaluation).
  3. ROUGE-L (F1):
       Longest Common Subsequence (LCS) F1 score over word tokens.
  4. Character Sequence Similarity:
       Character-level similarity ratio in [0.0, 1.0] via difflib.SequenceMatcher.
  5. Substring / Contains Match:
       Binary 1 if normalized reference is contained within student answer or vice-versa.

Outputs Generated:
  - results/gemini_scored_results.csv   (Per-question detailed scores)
  - results/gemini_leaderboard.csv      (Dataset-level aggregated summary)
  - results/gemini_evaluation_report.md (Formatted Markdown report)
  - Console summary report
"""

import csv
import difflib
import json
import os
import re
import string
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Force line-buffered UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR                = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR             = os.path.join(BASE_DIR, "results")
RAW_ANSWERS_PATH        = os.path.join(RESULTS_DIR, "gemini_raw_answers.json")
SCORED_RESULTS_PATH     = os.path.join(RESULTS_DIR, "gemini_scored_results.csv")
LEADERBOARD_PATH        = os.path.join(RESULTS_DIR, "gemini_leaderboard.csv")
REPORT_PATH             = os.path.join(RESULTS_DIR, "gemini_evaluation_report.md")

SUBJECT_TO_DATASET = {
    "science":                     "SciQ",
    "general_science":             "OpenBookQA",
    "science_challenge":           "ARC-Challenge",
    "reading_comprehension":       "RACE",
    "reading_comprehension_squad": "SQuAD v1.1",
}


# ── Text Normalization & Metrics ──────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """Standard QA normalization: lowercase, remove punctuation, remove articles, trim spaces."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles (a, an, the)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(reference: str, candidate: str) -> int:
    """Return 1 if normalised strings match exactly, else 0."""
    return 1 if normalise_text(reference) == normalise_text(candidate) else 0


def compute_token_f1(reference: str, candidate: str) -> Tuple[float, float, float]:
    """
    Computes token-level precision, recall, and F1 score.
    Returns (precision, recall, f1).
    """
    ref_tokens = normalise_text(reference).split()
    cand_tokens = normalise_text(candidate).split()

    if not ref_tokens and not cand_tokens:
        return 1.0, 1.0, 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0, 0.0, 0.0

    ref_counts: Dict[str, int] = defaultdict(int)
    for tok in ref_tokens:
        ref_counts[tok] += 1

    overlap = 0
    cand_counts: Dict[str, int] = defaultdict(int)
    for tok in cand_tokens:
        cand_counts[tok] += 1

    for tok, cnt in cand_counts.items():
        overlap += min(cnt, ref_counts.get(tok, 0))

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)


def _compute_lcs(x: List[str], y: List[str]) -> int:
    """Computes length of Longest Common Subsequence between two token sequences."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    Computes ROUGE-L F1 score based on word-level Longest Common Subsequence.
    """
    ref_tokens = normalise_text(reference).split()
    cand_tokens = normalise_text(candidate).split()

    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0

    lcs_len = _compute_lcs(ref_tokens, cand_tokens)
    rec = lcs_len / len(ref_tokens)
    prec = lcs_len / len(cand_tokens)
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(f1, 4)


def compute_char_similarity(reference: str, candidate: str) -> float:
    """Character-level SequenceMatcher similarity ratio in [0.0, 1.0]."""
    norm_ref = normalise_text(reference)
    norm_cand = normalise_text(candidate)
    if not norm_ref and not norm_cand:
        return 1.0
    return round(difflib.SequenceMatcher(None, norm_ref, norm_cand).ratio(), 4)


def compute_contains_match(reference: str, candidate: str) -> int:
    """Return 1 if reference is in candidate or candidate is in reference, else 0."""
    norm_ref = normalise_text(reference)
    norm_cand = normalise_text(candidate)
    if not norm_ref or not norm_cand:
        return 0
    return 1 if (norm_ref in norm_cand or norm_cand in norm_ref) else 0


# ── Scoring Engine ────────────────────────────────────────────────────────────

def score_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score each question item against the ground truth reference."""
    scored = []
    for rec in records:
        ref = rec.get("reference_answer", "") or ""
        cand = rec.get("gemini_answer", "") or ""
        model_used = rec.get("model_used", "none") or "none"

        em = compute_exact_match(ref, cand)
        prec, rec_score, f1 = compute_token_f1(ref, cand)
        rl = compute_rouge_l(ref, cand)
        char_sim = compute_char_similarity(ref, cand)
        contains = compute_contains_match(ref, cand)

        subject = rec.get("subject", "unknown")
        dataset = rec.get("source_dataset") or SUBJECT_TO_DATASET.get(subject, subject)

        scored.append({
            "model":            model_used,
            "id":               rec.get("id", ""),
            "source_dataset":   dataset,
            "subject":          subject,
            "question":         rec.get("question", ""),
            "reference_answer": ref,
            "gemini_answer":    cand,
            "exact_match":      em,
            "token_f1":         f1,
            "token_precision":  prec,
            "token_recall":     rec_score,
            "rouge_l":          rl,
            "char_similarity":  char_sim,
            "contains_match":   contains,
            "error":            rec.get("error", None),
        })
    return scored


def aggregate_by_dataset(scored_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate metrics grouped by dataset."""
    by_ds = defaultdict(list)
    for r in scored_records:
        by_ds[r["source_dataset"]].append(r)

    rows = []
    for ds_name, items in sorted(by_ds.items()):
        n = len(items)
        if n == 0:
            continue
        em_rate   = sum(x["exact_match"] for x in items) / n
        avg_f1    = sum(x["token_f1"] for x in items) / n
        avg_rl    = sum(x["rouge_l"] for x in items) / n
        avg_sim   = sum(x["char_similarity"] for x in items) / n
        cont_rate = sum(x["contains_match"] for x in items) / n
        models    = sorted(list(set(x["model"] for x in items)))

        rows.append({
            "dataset":             ds_name,
            "subject":             items[0]["subject"],
            "n_questions":         n,
            "exact_match_rate":    round(em_rate, 4),
            "avg_token_f1":        round(avg_f1, 4),
            "avg_rouge_l":         round(avg_rl, 4),
            "avg_char_similarity": round(avg_sim, 4),
            "contains_match_rate": round(cont_rate, 4),
            "models_used":         "|".join(models),
        })

    # Sort by exact match rate then F1 descending
    rows.sort(key=lambda x: (x["exact_match_rate"], x["avg_token_f1"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


# ── Writers & Reports ─────────────────────────────────────────────────────────

def save_scored_csv(scored_records: List[Dict[str, Any]], path: str):
    """Save item-level scored answers to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "model", "id", "source_dataset", "subject", "question",
        "reference_answer", "gemini_answer",
        "exact_match", "token_f1", "token_precision", "token_recall",
        "rouge_l", "char_similarity", "contains_match", "error"
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_records)
    print(f"[Save] Item-level scored results -> '{path}' ({len(scored_records)} rows)")


def save_leaderboard_csv(leaderboard_rows: List[Dict[str, Any]], path: str):
    """Save dataset summary metrics to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "rank", "dataset", "subject", "n_questions",
        "exact_match_rate", "avg_token_f1", "avg_rouge_l",
        "avg_char_similarity", "contains_match_rate", "models_used"
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard_rows)
    print(f"[Save] Dataset leaderboard       -> '{path}' ({len(leaderboard_rows)} rows)")


def generate_markdown_report(scored_records: List[Dict[str, Any]],
                             leaderboard_rows: List[Dict[str, Any]],
                             path: str):
    """Generate a clean Markdown summary report."""
    total_q = len(scored_records)
    overall_em = sum(r["exact_match"] for r in scored_records) / total_q if total_q else 0.0
    overall_f1 = sum(r["token_f1"] for r in scored_records) / total_q if total_q else 0.0
    overall_rl = sum(r["rouge_l"] for r in scored_records) / total_q if total_q else 0.0
    overall_sim = sum(r["char_similarity"] for r in scored_records) / total_q if total_q else 0.0
    overall_cont = sum(r["contains_match"] for r in scored_records) / total_q if total_q else 0.0

    lines = [
        "# Gemini Benchmark Evaluation Report",
        "",
        f"- **Source File**: `results/gemini_raw_answers.json`",
        f"- **Total Questions Evaluated**: {total_q}",
        f"- **Overall Exact Match (EM)**: {overall_em * 100:.1f}%",
        f"- **Overall Mean Token F1**: {overall_f1:.4f}",
        f"- **Overall Mean ROUGE-L**: {overall_rl:.4f}",
        f"- **Overall Mean Char Similarity**: {overall_sim:.4f}",
        f"- **Overall Contains Match Rate**: {overall_cont * 100:.1f}%",
        "",
        "## Performance by Dataset Leaderboard",
        "",
        "| Rank | Dataset | Subject | N | Exact Match | Token F1 | ROUGE-L | Char Sim | Contains Match | Models Used |",
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for row in leaderboard_rows:
        lines.append(
            f"| {row['rank']} | **{row['dataset']}** | `{row['subject']}` | {row['n_questions']} | "
            f"{row['exact_match_rate'] * 100:.1f}% | {row['avg_token_f1']:.4f} | {row['avg_rouge_l']:.4f} | "
            f"{row['avg_char_similarity']:.4f} | {row['contains_match_rate'] * 100:.1f}% | `{row['models_used']}` |"
        )

    lines.extend([
        "",
        "## Detailed Question-by-Question Breakdown",
        "",
        "| ID | Dataset | Reference Answer | Gemini Answer | EM | F1 | ROUGE-L | Sim |",
        "|:---|:---|:---|:---|:---:|:---:|:---:|:---:|",
    ])

    for r in scored_records:
        ref_esc = r["reference_answer"].replace("|", "\\|")
        cand_esc = (r["gemini_answer"] or "").replace("|", "\\|")
        lines.append(
            f"| `{r['id']}` | {r['source_dataset']} | {ref_esc} | {cand_esc} | "
            f"{'✅ 1' if r['exact_match'] else '❌ 0'} | {r['token_f1']:.2f} | {r['rouge_l']:.2f} | {r['char_similarity']:.2f} |"
        )

    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[Save] Markdown evaluation report -> '{path}'")


def print_console_summary(scored_records: List[Dict[str, Any]], leaderboard_rows: List[Dict[str, Any]]):
    """Print an ASCII leaderboard table to the console."""
    total_q = len(scored_records)
    overall_em = sum(r["exact_match"] for r in scored_records) / total_q if total_q else 0.0
    overall_f1 = sum(r["token_f1"] for r in scored_records) / total_q if total_q else 0.0
    overall_rl = sum(r["rouge_l"] for r in scored_records) / total_q if total_q else 0.0
    overall_sim = sum(r["char_similarity"] for r in scored_records) / total_q if total_q else 0.0
    overall_cont = sum(r["contains_match"] for r in scored_records) / total_q if total_q else 0.0

    print()
    print("=" * 86)
    print("  GEMINI BENCHMARK EVALUATION LEADERBOARD")
    print("=" * 86)
    print(f"  {'Rank':<5} {'Dataset':<16} {'Subject':<28} {'N':>3}  {'EM %':>7}  {'F1':>6}  {'ROUGE-L':>7}  {'Sim':>6}")
    print("-" * 86)
    for r in leaderboard_rows:
        print(
            f"  {r['rank']:<5} {r['dataset']:<16} {r['subject']:<28} {r['n_questions']:>3}  "
            f"{r['exact_match_rate']*100:>6.1f}%  {r['avg_token_f1']:>6.3f}  {r['avg_rouge_l']:>7.3f}  {r['avg_char_similarity']:>6.3f}"
        )
    print("=" * 86)
    print(
        f"  {'OVERALL':<51} {total_q:>3}  "
        f"{overall_em*100:>6.1f}%  {overall_f1:>6.3f}  {overall_rl:>7.3f}  {overall_sim:>6.3f}"
    )
    print(f"  (Contains / Substring Match Rate: {overall_cont * 100:.1f}%)")
    print("=" * 86)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RAW_ANSWERS_PATH):
        print(f"[Error] Raw answers file not found at: '{RAW_ANSWERS_PATH}'")
        sys.exit(1)

    with open(RAW_ANSWERS_PATH, encoding="utf-8") as fh:
        raw_records = json.load(fh)

    if not raw_records:
        print(f"[Error] No records found in '{RAW_ANSWERS_PATH}'")
        sys.exit(1)

    print(f"\n[Load] Loaded {len(raw_records)} records from '{RAW_ANSWERS_PATH}'")

    # 1. Score items
    scored = score_records(raw_records)

    # 2. Aggregate by dataset
    leaderboard = aggregate_by_dataset(scored)

    # 3. Save CSVs and Markdown Report
    save_scored_csv(scored, SCORED_RESULTS_PATH)
    save_leaderboard_csv(leaderboard, LEADERBOARD_PATH)
    generate_markdown_report(scored, leaderboard, REPORT_PATH)

    # 4. Display console summary
    print_console_summary(scored, leaderboard)


if __name__ == "__main__":
    main()
