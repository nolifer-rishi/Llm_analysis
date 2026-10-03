"""
EduBench-Local: Evaluator
==========================
Orchestrates all 3 metrics (ROUGE-L, BERTScore, LLM-Judge) and
produces the final scored_results.csv.
"""

import json
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from src.metrics.rouge_metric import compute_rouge_l_batch
from src.metrics.bertscore_metric import compute_bertscore_batch
from src.metrics.llm_judge import judge_batch


def load_raw_answers():
    """Load raw answers from the generation step."""
    path = config.RAW_ANSWERS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Raw answers not found at {path}. "
            "Run the answer generation step first."
        )
    
    with open(path, "r", encoding="utf-8") as f:
        answers = json.load(f)
    
    # Filter out errored answers
    valid = [a for a in answers if not a.get("error")]
    errored = len(answers) - len(valid)
    
    if errored > 0:
        print(f"  ⚠️  Skipping {errored} answers with errors")
    
    print(f"  Loaded {len(valid)} valid answers")
    return valid


def evaluate_all(answers=None, force=False):
    """
    Run all 3 metrics on the generated answers.
    
    Args:
        answers: list of answer dicts (if None, loads from file)
        force: If True, recompute all metrics
        
    Returns:
        pd.DataFrame with all scores
    """
    if answers is None:
        answers = load_raw_answers()
    
    # Check if results already exist
    if not force and config.SCORED_RESULTS_FILE.exists():
        print(f"  Scored results already exist at {config.SCORED_RESULTS_FILE}")
        print(f"  Loading existing results (use --force to recompute)")
        return pd.read_csv(config.SCORED_RESULTS_FILE)
    
    references = [a["reference_answer"] for a in answers]
    generated = [a["generated_answer"] for a in answers]
    
    # ── Metric 1: ROUGE-L ────────────────────────────────────────────
    print("\n[1/3] Computing ROUGE-L scores...")
    rouge_scores = compute_rouge_l_batch(references, generated)
    print(f"  ✓ ROUGE-L computed ({len(rouge_scores)} scores)")
    print(f"  Mean ROUGE-L: {sum(rouge_scores) / len(rouge_scores):.4f}")
    
    # ── Metric 2: BERTScore ──────────────────────────────────────────
    print("\n[2/3] Computing BERTScore F1...")
    bert_scores = compute_bertscore_batch(references, generated)
    print(f"  ✓ BERTScore computed ({len(bert_scores)} scores)")
    print(f"  Mean BERTScore F1: {sum(bert_scores) / len(bert_scores):.4f}")
    
    # ── Metric 3: LLM-as-Judge ──────────────────────────────────────
    print("\n[3/3] Computing LLM-Judge scores (limited to 10 per dataset)...")
    judge_answers = []
    counts = {}
    for a in answers:
        ds = a["dataset"]
        if counts.get(ds, 0) < 10:
            judge_answers.append(a)
            counts[ds] = counts.get(ds, 0) + 1
            
    judge_results = judge_batch(judge_answers, force=force)
    
    # Extract judge scores aligned with answers
    judge_scores = []
    for a in answers:
        result = judge_results.get(a["id"], {})
        score = result.get("score", None)
        judge_scores.append(score)
    
    valid_judge = [s for s in judge_scores if s is not None]
    if valid_judge:
        print(f"  Mean Judge Score: {sum(valid_judge) / len(valid_judge):.2f}")
    
    # ── Build DataFrame ──────────────────────────────────────────────
    print("\nBuilding results DataFrame...")
    
    rows = []
    for i, a in enumerate(answers):
        row = {
            "id": a["id"],
            "dataset": a["dataset"],
            "subject": a["subject"],
            "question_type": a["question_type"],
            "question": a["question"],
            "reference_answer": a["reference_answer"],
            "generated_answer": a["generated_answer"],
            "model": a["model"],
            "rouge_l": rouge_scores[i],
            "bertscore_f1": bert_scores[i],
            "judge_score": judge_scores[i],
            "latency_sec": a["latency_sec"],
            "token_count": a["token_count"],
        }
        
        # Add MCQ accuracy for MCQ questions
        if a["question_type"] == "mcq" and a.get("answer_key"):
            row["answer_key"] = a["answer_key"]
            row["mcq_correct"] = _check_mcq_correctness(
                a["generated_answer"], a["answer_key"]
            )
        else:
            row["answer_key"] = None
            row["mcq_correct"] = None
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    config.SCORED_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SCORED_RESULTS_FILE, index=False)
    print(f"\n✓ Saved scored results to {config.SCORED_RESULTS_FILE}")
    print(f"  Shape: {df.shape}")
    
    return df


def _check_mcq_correctness(generated_answer, correct_key):
    """
    Check if the generated MCQ answer matches the correct answer key.
    Looks for the letter at the start of the response or after common prefixes.
    """
    if not generated_answer or not correct_key:
        return None
    
    generated = generated_answer.strip().upper()
    correct = correct_key.strip().upper()
    
    # Check if the response starts with the correct letter
    if generated.startswith(correct):
        return True
    
    # Check common patterns: "The answer is B", "B)", "B."
    import re
    patterns = [
        rf'\b{correct}\b',          # Standalone letter
        rf'^{correct}[).\s]',        # Letter at start with delimiter
        rf'answer\s+is\s+{correct}', # "answer is X"
        rf'correct\s+answer\s+is\s+{correct}',
    ]
    
    for pattern in patterns:
        if re.search(pattern, generated, re.IGNORECASE):
            return True
    
    # Check if any wrong letter appears first
    for letter in "ABCDE":
        if letter != correct and generated.startswith(letter):
            return False
    
    return False  # Default to incorrect if we can't determine


if __name__ == "__main__":
    force = "--force" in sys.argv
    df = evaluate_all(force=force)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df[["dataset", "rouge_l", "bertscore_f1", "judge_score"]].describe())
