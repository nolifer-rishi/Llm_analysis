# EduBench-Local: Comparative Performance Analysis of Open-Source LLMs for Educational Question Answering

**EduBench-Local** is a fully offline, cost-free benchmark pipeline designed for student research to compare open-source Large Language Models (LLMs) on educational question answering across subjects using a tri-metric evaluation system.

---

## 🏗️ Architecture & Pipeline Overview

```
                          ┌──────────────────────────┐
                          │   Hugging Face Datasets  │
                          │     (SciQ & SQuAD/ARC)   │
                          └─────────────┬────────────┘
                                        │
                             [prepare_datasets.py]
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   dataset_sample.json    │
                          └─────────────┬────────────┘
                                        │
                             [generate_answers.py]  ◄──  Ollama Local LLMs
                                        │               (Llama 3.2, Mistral, Gemma 2, etc.)
                                        ▼
                          ┌──────────────────────────┐
                          │     raw_answers.json     │
                          └─────────────┬────────────┘
                                        │
                             [evaluate_metrics.py]
                             ├── ROUGE-L (Lexical)
                             ├── BERTScore (Semantic)
                             └── LLM-as-Judge (Quality: 1-5)
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │    scored_results.csv    │
                          └─────────────┬────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
          [analyze_and_plot.py]                         [app.py]
   ├── Leaderboard & Combined Scores             Streamlit Interactive Dashboard
   ├── Subject Heatmaps & Radars                 ├── Leaderboard & KPIs
   ├── Efficiency Frontier                       ├── Qualitative Answer Inspector
   └── Wilcoxon Significance Tests               └── Side-by-Side Model Comparison
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```powershell
.\edubench-env\Scripts\activate
```

### 2. Prepare Datasets (SciQ + Reading Comprehension)
```powershell
python scripts/prepare_datasets.py --per-dataset 150 --seed 42
```
*Creates all origin files and `data/dataset_sample.json` (750 questions).*

### 3. Generate Answers Across Models
Pull any target models in Ollama:
```powershell
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull gemma2:2b
ollama pull phi3:mini
ollama pull qwen2.5:3b
```

Run generation:
```powershell
python scripts/generate_answers.py --dataset data/dataset_sample.json --models llama3.2:3b mistral:7b gemma2:2b phi3:mini qwen2.5:3b
```
*(Supports checkpointing — resuming will not re-generate completed answers).*

### 4. Tri-Metric Evaluation
```powershell
python scripts/evaluate_metrics.py --input results/raw_answers.json --output-csv results/scored_results.csv --judge-model llama3.2:3b
```

### 5. Aggregation, Statistical Analysis & Visualizations
```powershell
python scripts/analyze_and_plot.py --input results/scored_results.csv --output-dir plots
```
Generates:
- `plots/leaderboard_bar.png`
- `plots/subject_heatmap.png`
- `plots/radar_metrics.png`
- `plots/efficiency_frontier.png`
- `plots/leaderboard_summary.csv`
- `plots/wilcoxon_p_values.csv`

### 6. Launch Interactive Dashboard
```powershell
streamlit run app.py
```

---

## 📐 The Three Evaluation Metrics

| Metric | Category | Dimension Captured | Library / Backend |
|---|---|---|---|
| **ROUGE-L** | Lexical | Surface word & phrase overlap with reference | `rouge-score` |
| **BERTScore (F1)** | Semantic | Contextual embedding similarity (paraphrase-robust) | `bert-score` (`roberta-large`) |
| **LLM-as-Judge (1–5)** | Correctness / Quality | Factual accuracy and conceptual completeness | Local Ollama Model |

---

## 📁 File Descriptions

### 🐍 Scripts (`scripts/`)

| File | Description |
|---|---|
| `prepare_datasets.py` | Downloads and standardizes 5 educational QA datasets from HuggingFace (SciQ, ARC-Challenge, OpenBookQA, RACE, SQuAD). Samples 150 questions per dataset, saves individual origin files, subject-domain subsets, and the master 750-question `dataset_sample.json`. Supports `--focused` flag to build the SciQ × RACE 300Q cross-domain benchmark only. |
| `generate_answers.py` | Loads standardized questions and queries local LLMs via Ollama using a fixed pedagogical prompt template. Tracks latency and token counts per answer. Saves results incrementally to `raw_answers.json` with checkpointing support (skips already-generated answers on resume). |
| `evaluate_metrics.py` | Tri-metric evaluation engine. Computes ROUGE-L (lexical overlap), BERTScore F1 (semantic similarity via `roberta-large`), and LLM-as-Judge scores (1–5 scale, greedy decoding). Outputs composite normalized score and saves `scored_results.csv` + `scored_results.json`. |
| `analyze_and_plot.py` | Aggregation, ranking, and visualization engine. Produces normalized leaderboard rankings, dataset-wise performance breakdowns, pairwise Wilcoxon signed-rank tests, and 5 publication-ready figures saved to `plots/`. |
| `cross_domain_analysis.py` | Flagship cross-domain analysis comparing Science (SciQ) vs Reading Comprehension (RACE). Computes Mann-Whitney U tests, Cohen's d effect sizes, 95% CIs, and generates 4 specialized figures: grouped bar, violin distribution, BERTScore vs Judge scatter, and radar chart. |
| `validate_judge.py` | Human validation tool for the LLM-as-Judge metric. Interactively prompts a human evaluator to rate sampled answers (1–5), then computes Cohen's Kappa, Pearson correlation, MAE, and exact-match agreement. Saves results to `plots/human_validation_agreement.csv`. |
| `build_focused_benchmark.py` | Lightweight standalone script to build only the focused 300-question SciQ × RACE cross-domain benchmark from cached origin files. |

---

### 📊 Data Files (`data/`)

| File | Description |
|---|---|
| `dataset_sample.json` | **Master 750-question benchmark.** Combines all 5 datasets (150 questions each) in a unified schema with fields: `id`, `source_dataset`, `subject`, `question`, `context`, `reference_answer`, `options`. |
| `sciq_origin.json` | 150 SciQ science exam questions with textbook support passages and 4 answer choices. Evidence-based factual recall. |
| `arc_challenge_origin.json` | 150 ARC-Challenge grade-school science reasoning questions. Zero-shot, no supporting context — tests multi-step deduction. |
| `openbookqa_origin.json` | 150 OpenBookQA elementary science questions. Tests commonsense scientific reasoning without passage context. |
| `race_reading_origin.json` | 150 RACE middle/high school English reading comprehension questions with full article passages. Tests literary inference and comprehension. |
| `squad_reading_origin.json` | 150 SQuAD v1.1 reading comprehension questions with Wikipedia passage context. Tests information extraction and synthesis. |
| `science_domain_origin.json` | Combined 450-question science domain subset (SciQ + ARC + OpenBookQA). |
| `reading_domain_origin.json` | Combined 300-question reading comprehension domain subset (RACE + SQuAD). |
| `sciq_race_benchmark.json` | Focused 300-question cross-domain benchmark (SciQ × RACE) with explicit `domain` field for downstream cross-domain analysis. |

---

### 📈 Results (`results/`)

| File | Description |
|---|---|
| `raw_answers.json` | Raw model outputs from `generate_answers.py`. Each record contains: `id`, `model`, `question`, `context`, `reference_answer`, `generated_answer`, `prompt`, `latency_sec`, `token_count`, `timestamp`. |
| `scored_results.json` | Fully evaluated dataset with all metric scores. Adds `rouge_l`, `bertscore_f1`, `judge_score`, `normalized_judge_score`, `combined_score`, `source_dataset` to every record. |
| `scored_results.csv` | Flat CSV version of `scored_results.json` — ready for Excel, pandas, or any statistical tool. |

---

### 🖼️ Plots & Outputs (`plots/`)

| File | Description |
|---|---|
| `leaderboard_bar.png` | Horizontal grouped bar chart ranking all models by ROUGE-L, BERTScore, and LLM-Judge scores side-by-side. |
| `subject_heatmap.png` | Two-panel heatmap: (1) Dataset × Metric performance grid, (2) Model × Dataset performance breakdown. |
| `dataset_breakdown.png` | Bar chart showing per-dataset performance breakdown across all three metrics for each model. |
| `radar_metrics.png` | Radar/spider chart visualizing multi-dimensional metric profiles per model across ROUGE-L, BERTScore, and Judge Score. |
| `efficiency_frontier.png` | Scatter plot of Combined Score vs. Average Latency (s) — identifies the best speed-accuracy trade-off models. |
| `leaderboard_summary.csv` | Aggregated model leaderboard with mean scores for all three metrics, combined score, and ranking. |
| `dataset_summary.csv` | Per-dataset aggregated statistics across all evaluated models. |
| `subject_summary.csv` | Domain-level (science vs. reading comprehension) aggregated performance summary. |
| `human_validation_agreement.csv` | Inter-rater agreement records from `validate_judge.py`: human scores, LLM-judge scores, and absolute differences per sampled item. |

---

### 📄 Root Files

| File | Description |
|---|---|
| `app.py` | Streamlit interactive research dashboard. Features: KPI cards, model leaderboard table, per-dataset metric breakdowns, qualitative answer inspector with filtering, side-by-side model comparison, and embedded publication plots. Dark-mode themed with Inter font. |
| `research_paper.md` | Full academic research paper in Markdown format covering abstract, introduction, related work, dataset suite, methodology, experimental results, discussion, human validation, and references. |
| `README.md` | This file — project overview, pipeline architecture, quickstart guide, metric descriptions, and complete file reference. |
| `.gitignore` | Excludes Python caches, virtual environment (`edubench-env/`), HuggingFace model weights, logs, and IDE files from version control. |
| `.streamlit/config.toml` | Streamlit theme configuration — dark mode with indigo primary color (`#6366F1`), dark background (`#0B0F19`), and Inter font. |

---

## 📊 Key Results (Llama 3.2 3B — 750 Questions)

| Dataset | Judge Score (1–5) | BERTScore F1 | ROUGE-L F1 | Avg Latency |
|---|---|---|---|---|
| **SciQ** | **4.38** 🥇 | 0.819 | 0.052 | 1.11s |
| **SQuAD** | 4.07 | 0.854 | **0.150** 🥇 | **0.87s** ⚡ |
| **ARC-Challenge** | 4.03 | 0.841 | 0.064 | 1.10s |
| **OpenBookQA** | 4.03 | 0.821 | 0.034 | 1.02s |
| **RACE** | 3.92 | **0.864** 🥇 | 0.123 | 0.98s |
| **Overall** | **4.09 ± 0.46** | **0.840 ± 0.05** | **0.084 ± 0.11** | **1.01s** |

---

## 📁 Repository Structure

```
edubench-project/
├── app.py                          # Streamlit interactive dashboard
├── research_paper.md               # Full academic research paper
├── README.md                       # Project overview & documentation
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   └── config.toml                 # Streamlit dark theme config
├── scripts/
│   ├── prepare_datasets.py         # Dataset download & standardization
│   ├── generate_answers.py         # Multi-model LLM inference
│   ├── evaluate_metrics.py         # ROUGE-L, BERTScore, LLM-Judge scoring
│   ├── analyze_and_plot.py         # Aggregation, stats & visualizations
│   ├── cross_domain_analysis.py    # SciQ × RACE cross-domain analysis
│   ├── validate_judge.py           # Human vs LLM-Judge agreement tool
│   └── build_focused_benchmark.py  # 300Q focused benchmark builder
├── data/
│   ├── dataset_sample.json         # Master 750-question benchmark
│   ├── sciq_origin.json            # 150 SciQ science questions
│   ├── arc_challenge_origin.json   # 150 ARC-Challenge questions
│   ├── openbookqa_origin.json      # 150 OpenBookQA questions
│   ├── race_reading_origin.json    # 150 RACE reading questions
│   ├── squad_reading_origin.json   # 150 SQuAD questions
│   ├── science_domain_origin.json  # 450 combined science questions
│   ├── reading_domain_origin.json  # 300 combined reading questions
│   └── sciq_race_benchmark.json    # 300Q focused cross-domain benchmark
├── results/
│   ├── raw_answers.json            # Raw LLM outputs with metadata
│   ├── scored_results.json         # Evaluated results with all metrics
│   └── scored_results.csv          # Flat CSV for analysis tools
└── plots/
    ├── leaderboard_bar.png         # Model leaderboard bar chart
    ├── subject_heatmap.png         # Dataset × Metric heatmap
    ├── dataset_breakdown.png       # Per-dataset metric breakdown
    ├── radar_metrics.png           # Radar metric profile chart
    ├── efficiency_frontier.png     # Speed vs accuracy scatter
    ├── leaderboard_summary.csv     # Aggregated leaderboard data
    ├── dataset_summary.csv         # Per-dataset statistics
    ├── subject_summary.csv         # Domain-level statistics
    └── human_validation_agreement.csv  # Human vs judge agreement
```