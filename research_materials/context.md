# EduBench-Local — Session Context

## Current Phase
Phase 10: Execution — Full answer generation running

## What's Been Completed
- **Phase 1**: config.py, requirements.txt, __init__.py files, context.md, workdone.md
- **Phase 2**: src/dataset_loader.py — all 6 datasets loaded & standardized
- **Phase 3**: src/prompt_builder.py — open-ended + MCQ templates
- **Phase 4**: src/answer_generator.py — Ollama generation with checkpointing
- **Phase 5**: src/metrics/ — ROUGE-L, BERTScore, LLM-as-Judge
- **Phase 6**: src/evaluator.py — orchestrates all 3 metrics
- **Phase 7**: src/aggregator.py + src/statistics.py — ranking + stat tests
- **Phase 8**: src/visualizer.py — 7 publication-quality figure types
- **Phase 9**: run_pipeline.py — CLI entry point
- **Dependencies**: All pip packages installed
- **Model**: Mistral 7B pulled (4.4 GB)
- **Datasets**: 600 questions sampled (100 per dataset) saved to data/processed/dataset_sample.json
- **Dry run**: 6 questions tested successfully, ~14s/question, 0 errors

## What's In Progress
- Full answer generation (resumed) — 380/600 completed, 220 remaining.
- Estimated completion in ~40 minutes.
- Checkpoints saved every 10 questions to results/raw_answers.json

## What's Next (After Generation)
1. Run evaluation step (ROUGE-L + BERTScore + LLM-Judge)
2. Run aggregation + statistical analysis
3. Run visualization

## Configuration State
- Model: mistral:7b
- Datasets: SciQ, OpenBookQA, ARC-Easy, ARC-Challenge, RACE, SQuAD (100 each)
- Metrics: ROUGE-L, BERTScore F1, LLM-as-Judge (1-5)
- GPU: CPU-only (no NVIDIA GPU), ~14s/question

## Known Issues
- Windows cp1252 encoding — use $env:PYTHONIOENCODING="utf-8" when running
- trust_remote_code warnings from HuggingFace (harmless)
- Self-judging bias (Mistral judges itself) — acknowledged limitation

## Last Updated
2026-09-17 22:58 IST
