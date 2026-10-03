# DeepSeek-R1:7b — EduBench-Local Analysis

Isolated benchmark run of **deepseek-r1:7b** across the full 750-question EduBench-Local suite
(SciQ 150 + ARC-Challenge 150 + OpenBookQA 150 + RACE 150 + SQuAD 150).

## Folder Structure

```
deepseek-analysis/
├── scripts/
│   ├── generate_deepseek_answers.py   # Step 1: Generate answers
│   ├── evaluate_deepseek.py           # Step 2: Score with ROUGE-L, BERTScore, LLM-Judge
│   └── analyze_deepseek.py            # Step 3: Plots & report
├── results/
│   ├── deepseek_raw_answers.json      # Raw + cleaned answers (auto-generated)
│   ├── deepseek_scored_results.json   # Scored results (auto-generated)
│   └── deepseek_scored_results.csv    # Scored results CSV (auto-generated)
└── plots/
    ├── deepseek_by_dataset.png        # Per-dataset metric bars
    ├── deepseek_score_dist.png        # Score histograms
    ├── deepseek_latency.png           # Latency distribution
    └── deepseek_think_impact.png      # <think> block vs no-think comparison
```

## Key Feature: <think> Tag Handling

DeepSeek-R1 emits chain-of-thought inside `<think>...</think>` blocks.
This pipeline automatically strips them before metric scoring — but **preserves
the raw output** in `raw_answer` field for reference.

## How to Run (from project root)

```powershell
# Activate environment
.\edubench-env\Scripts\Activate.ps1

# Step 1 — Generate answers (resume-safe, ~4–8 hours for 750 Qs)
python deepseek-analysis\scripts\generate_deepseek_answers.py

# Step 2 — Score (ROUGE-L + BERTScore + LLM-Judge)
python deepseek-analysis\scripts\evaluate_deepseek.py

# Step 3 — Plots & report
python deepseek-analysis\scripts\analyze_deepseek.py
```

### Quick test (5 questions):
```powershell
python deepseek-analysis\scripts\generate_deepseek_answers.py --limit 5
```
