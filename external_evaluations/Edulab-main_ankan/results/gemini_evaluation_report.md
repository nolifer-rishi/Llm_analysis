# Gemini Benchmark Evaluation Report

- **Source File**: `results/gemini_raw_answers.json`
- **Total Questions Evaluated**: 5
- **Overall Exact Match (EM)**: 40.0%
- **Overall Mean Token F1**: 0.5333
- **Overall Mean ROUGE-L**: 0.5333
- **Overall Mean Char Similarity**: 0.5697
- **Overall Contains Match Rate**: 60.0%

## Performance by Dataset Leaderboard

| Rank | Dataset | Subject | N | Exact Match | Token F1 | ROUGE-L | Char Sim | Contains Match | Models Used |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **ARC-Challenge** | `science_challenge` | 1 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 100.0% | `gemini-3.6-flash` |
| 2 | **SciQ** | `science` | 1 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 100.0% | `gemini-3.6-flash` |
| 3 | **SQuAD v1.1** | `reading_comprehension_squad` | 1 | 0.0% | 0.6667 | 0.6667 | 0.5714 | 100.0% | `gemini-3.6-flash` |
| 4 | **OpenBookQA** | `general_science` | 1 | 0.0% | 0.0000 | 0.0000 | 0.0769 | 0.0% | `gemini-3.8-flash` |
| 5 | **RACE** | `reading_comprehension` | 1 | 0.0% | 0.0000 | 0.0000 | 0.2000 | 0.0% | `gemini-3.6-flash` |

## Detailed Question-by-Question Breakdown

| ID | Dataset | Reference Answer | Gemini Answer | EM | F1 | ROUGE-L | Sim |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| `openbookqa_327` | OpenBookQA | in spiked plants | from cacti. | ❌ 0 | 0.00 | 0.00 | 0.08 |
| `race_middle_228` | RACE | she liked the little boy | love and kindness (or kindness and warmth / | ❌ 0 | 0.00 | 0.00 | 0.20 |
| `squad_228` | SQuAD v1.1 | 3,837 | 3,837 yards | ❌ 0 | 0.67 | 0.67 | 0.57 |
| `sciq_228` | SciQ | ethology | Ethology | ✅ 1 | 1.00 | 1.00 | 1.00 |
| `arc_challenge_228` | ARC-Challenge | endothermic. | Endothermic | ✅ 1 | 1.00 | 1.00 | 1.00 |
