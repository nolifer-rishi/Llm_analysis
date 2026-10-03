"""
EduBench-Local — Tri-Metric Evaluation Engine
---------------------------------------------
Evaluates generated answers against ground-truth references using three complementary metrics:
  1. Lexical Overlap: ROUGE-L (F1 measure)
  2. Semantic Similarity: BERTScore (F1 measure)
  3. Correctness / Quality: Local LLM-as-Judge (1-5 scale)

Usage:
  python evaluate_metrics.py [--input raw_answers.json] [--output scored_results.csv] [--judge-model llama3.2:3b] [--skip-judge]
"""

import os
import re
import json
import argparse
import pandas as pd
import ollama
from rouge_score import rouge_scorer
from bert_score import score as bert_score

JUDGE_PROMPT_TEMPLATE = """You are grading a student-facing AI tutor's answer.
Question: {question}
Reference (correct) answer: {reference}
AI-generated answer: {generated}

Rate the AI-generated answer from 1 to 5 for factual correctness and completeness relative to the reference answer:
1 = Completely incorrect or irrelevant
2 = Mostly incorrect or major factual errors
3 = Partially correct but missing key points
4 = Mostly correct and accurate with minor omissions
5 = Fully correct, comprehensive, and accurate

Respond with ONLY a single integer from 1 to 5."""


def evaluate_rouge_l(results: list[dict]) -> list[float]:
    """Computes ROUGE-L F1 scores for each generated answer."""
    print("[*] Computing ROUGE-L (lexical overlap)...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for item in results:
        ref = item.get("reference_answer", "")
        gen = item.get("generated_answer", "")
        score = scorer.score(ref, gen)
        f_measure = round(score["rougeL"].fmeasure, 4)
        scores.append(f_measure)
    print(f"[+] ROUGE-L computed for {len(scores)} answers. Mean: {sum(scores)/len(scores):.4f}")
    return scores


def evaluate_bertscore(results: list[dict], batch_size: int = 32) -> list[float]:
    """Computes BERTScore F1 scores for semantic similarity."""
    print("[*] Computing BERTScore (semantic similarity using contextual embeddings)...")
    references = [(item.get("reference_answer") or "").strip() or "No answer." for item in results]
    candidates = [(item.get("generated_answer") or "").strip() or "No answer provided." for item in results]
    
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False, batch_size=batch_size)
    f1_list = [round(score.item(), 4) for score in F1]
    print(f"[+] BERTScore computed for {len(f1_list)} answers. Mean: {sum(f1_list)/len(f1_list):.4f}")
    return f1_list


def evaluate_llm_judge(results: list[dict], judge_model: str = "llama3.2:3b") -> list[int]:
    """Uses a local LLM to grade answer correctness & completeness on a 1-5 integer scale."""
    print(f"[*] Computing LLM-as-Judge scores using model '{judge_model}'...")
    judge_scores = []

    for i, item in enumerate(results):
        question = item.get("question", "")
        reference = item.get("reference_answer", "")
        generated = item.get("generated_answer", "")
        model_name = item.get("model", "unknown")

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            reference=reference,
            generated=generated
        )

        score = None
        for attempt in range(2):
            try:
                response = ollama.chat(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0, "num_predict": 16}  # greedy decoding for deterministic grading
                )
                text = response["message"]["content"].strip()
                # Find digits between 1 and 5
                matches = re.findall(r"\b([1-5])\b", text)
                if matches:
                    score = int(matches[0])
                    break
                else:
                    # Fallback search for any first digit
                    digits = [int(c) for c in text if c in "12345"]
                    if digits:
                        score = digits[0]
                        break
            except Exception as e:
                print(f"    [!] Error querying judge model on item {i}: {e}")

        # Default fallback to median 3 if parsing completely failed
        if score is None:
            score = 3

        judge_scores.append(score)
        if (i + 1) % 5 == 0 or (i + 1) == len(results):
            print(f"    Graded [{i+1}/{len(results)}] | Model: {model_name} | Judge Score: {score}/5")

    print(f"[+] LLM-Judge complete for {len(judge_scores)} items. Mean: {sum(judge_scores)/len(judge_scores):.2f}/5")
    return judge_scores


def main():
    parser = argparse.ArgumentParser(description="Tri-Metric Evaluation Engine for EduBench-Local")
    parser.add_argument("--input", type=str, default="raw_answers.json", help="Path to raw generated answers JSON")
    parser.add_argument("--output-csv", type=str, default="scored_results.csv", help="Path for scored output CSV")
    parser.add_argument("--output-json", type=str, default="scored_results.json", help="Path for scored output JSON")
    parser.add_argument("--judge-model", type=str, default="llama3.2:3b", help="Ollama model to use as judge")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge metric (ROUGE & BERTScore only)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input file '{args.input}' not found. Run generate_answers.py first.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"[*] Loaded {len(results)} generated answers from {args.input}")

    # Compute Metric 1: ROUGE-L
    rouge_scores = evaluate_rouge_l(results)
    for i, item in enumerate(results):
        item["rouge_l"] = rouge_scores[i]

    # Compute Metric 2: BERTScore
    bert_scores = evaluate_bertscore(results)
    for i, item in enumerate(results):
        item["bertscore_f1"] = bert_scores[i]

    # Compute Metric 3: LLM-as-Judge
    if not args.skip_judge:
        judge_scores = evaluate_llm_judge(results, judge_model=args.judge_model)
        for i, item in enumerate(results):
            item["judge_score"] = judge_scores[i]
    else:
        for item in results:
            item["judge_score"] = 0

    # Compute Composite Scores (Normalized Judge & Tri-Metric Combined Score)
    for item in results:
        # Determine source_dataset if missing
        if "source_dataset" not in item or item["source_dataset"] == "General":
            qid = item.get("id", "")
            if qid.startswith("sciq"):
                item["source_dataset"] = "SciQ"
            elif qid.startswith("arc"):
                item["source_dataset"] = "ARC-Challenge"
            elif qid.startswith("obqa") or qid.startswith("openbook"):
                item["source_dataset"] = "OpenBookQA"
            elif qid.startswith("race"):
                item["source_dataset"] = "RACE"
            elif qid.startswith("squad"):
                item["source_dataset"] = "SQuAD"
            else:
                item["source_dataset"] = "General"

        # Normalize 1-5 judge score to [0, 1] range: 1 -> 0.0, 5 -> 1.0
        judge_val = float(item.get("judge_score", 0))
        norm_judge = max(0.0, min(1.0, (judge_val - 1.0) / 4.0)) if judge_val > 0 else 0.0
        item["normalized_judge_score"] = round(norm_judge, 4)

        # Composite tri-metric score = arithmetic mean of (ROUGE-L + BERTScore F1 + Normalized Judge)
        rouge_val = float(item.get("rouge_l", 0.0))
        bert_val = float(item.get("bertscore_f1", 0.0))
        combined = (rouge_val + bert_val + norm_judge) / 3.0
        item["combined_score"] = round(combined, 4)

    # Save to JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved scored results to JSON: {args.output_json}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[+] Saved scored results to CSV: {args.output_csv}")

    # Display preview summary table
    summary = df.groupby("model").agg(
        total_answers=("id", "count"),
        avg_rouge_l=("rouge_l", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
        avg_judge=("judge_score", "mean"),
        avg_combined=("combined_score", "mean"),
        avg_latency=("latency_sec", "mean")
    ).reset_index()

    print("\n" + "="*85)
    print("SCORED BENCHMARK SUMMARY (PREVIEW)")
    print("="*85)
    print(summary.to_string(index=False))
    print("="*85 + "\n")


if __name__ == "__main__":
    main()
