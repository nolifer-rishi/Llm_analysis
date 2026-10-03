"""
EduBench-Local — Step 2: Scoring Script
-----------------------------------------
Purpose: Load the results from step1_test_generation.py (test_results.json)
and score each generated answer against its reference answer using:
  1. ROUGE-L (lexical overlap)
  2. BERTScore (semantic similarity)

Run this AFTER step1_test_generation.py, from your activated virtual environment:
    python step2_scoring.py
"""

import json
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# -------------------------------------------------------------------
# 1. Load the results from Step 1
# -------------------------------------------------------------------
print("Loading test_results.json...")
with open("test_results.json", "r") as f:
    results = json.load(f)

references = [r["reference_answer"] for r in results]
generated = [r["generated_answer"] for r in results]

# -------------------------------------------------------------------
# 2. ROUGE-L scoring
# -------------------------------------------------------------------
print("Computing ROUGE-L scores...")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

for r in results:
    score = rouge.score(r["reference_answer"], r["generated_answer"])
    r["rouge_l"] = score["rougeL"].fmeasure

# -------------------------------------------------------------------
# 3. BERTScore scoring (this downloads a small model the first time)
# -------------------------------------------------------------------
print("Computing BERTScore (this may take a minute the first time)...")
P, R, F1 = bert_score(generated, references, lang="en", verbose=False)

for i, r in enumerate(results):
    r["bertscore_f1"] = F1[i].item()

# -------------------------------------------------------------------
# 4. Print a clean summary table
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"{'Question':<40} {'ROUGE-L':<10} {'BERTScore':<10}")
print("=" * 80)
for r in results:
    q_short = (r["question"][:37] + "...") if len(r["question"]) > 40 else r["question"]
    print(f"{q_short:<40} {r['rouge_l']:<10.3f} {r['bertscore_f1']:<10.3f}")

avg_rouge = sum(r["rouge_l"] for r in results) / len(results)
avg_bert = sum(r["bertscore_f1"] for r in results) / len(results)
print("=" * 80)
print(f"{'AVERAGE':<40} {avg_rouge:<10.3f} {avg_bert:<10.3f}")

# -------------------------------------------------------------------
# 5. Save the scored results
# -------------------------------------------------------------------
with open("scored_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone! Scored results saved to scored_results.json")
