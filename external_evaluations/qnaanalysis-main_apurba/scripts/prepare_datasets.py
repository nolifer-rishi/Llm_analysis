"""
EduBench-Local — Multi-Dataset Preparation Module
-------------------------------------------------
Downloads, standardizes, and saves educational QA datasets with explicit origin filenames:

Origin Files Generated in data/:
  - data/sciq_origin.json (Science Exam QA)
  - data/arc_challenge_origin.json (Grade-School Science Reasoning)
  - data/openbookqa_origin.json (Elementary Multi-step Science Reasoning)
  - data/race_reading_origin.json (Middle/High School English Reading Comprehension)
  - data/squad_reading_origin.json (Passage-grounded Reading Comprehension)
  - data/dataset_sample.json (Master Combined 750-Question Benchmark)
  - data/sciq_race_benchmark.json (Focused 300-Question Cross-Domain Benchmark: SciQ × RACE)

Usage:
  python scripts/prepare_datasets.py [--per-dataset 150] [--seed 42] [--output-dir data]
  python scripts/prepare_datasets.py --focused   # Build only the SciQ × RACE 300Q benchmark
"""

import os
import ast
import json
import random
import argparse
from datasets import load_dataset


def load_sciq(sample_size: int = 150, seed: int = 42) -> list[dict]:
    """Loads SciQ dataset."""
    print(f"[*] Loading SciQ ({sample_size} questions)...")
    data = []
    try:
        ds = load_dataset("allenai/sciq", split="test")
        sampled = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
        for i, item in enumerate(sampled):
            data.append({
                "id": f"sciq_{i:04d}",
                "source_dataset": "SciQ",
                "subject": "science",
                "question": item["question"].strip(),
                "context": item.get("support", "").strip(),
                "reference_answer": item["correct_answer"].strip(),
                "options": [
                    item["correct_answer"].strip(),
                    item.get("distractor1", "").strip(),
                    item.get("distractor2", "").strip(),
                    item.get("distractor3", "").strip(),
                ]
            })
        print(f"[+] Loaded {len(data)} from SciQ")
    except Exception as e:
        print(f"[!] SciQ load error: {e}")
    return data


def load_arc(sample_size: int = 150, seed: int = 42) -> list[dict]:
    """Loads ARC (AI2 Science Questions)."""
    print(f"[*] Loading ARC-Challenge ({sample_size} questions)...")
    data = []
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        sampled = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
        for i, item in enumerate(sampled):
            choices = item["choices"]["text"]
            key = item["answerKey"]
            try:
                idx = item["choices"]["label"].index(key)
                ref_ans = choices[idx]
            except Exception:
                ref_ans = choices[0] if choices else key

            data.append({
                "id": f"arc_{i:04d}",
                "source_dataset": "ARC-Challenge",
                "subject": "science",
                "question": item["question"].strip(),
                "context": "",
                "reference_answer": ref_ans.strip(),
                "options": choices
            })
        print(f"[+] Loaded {len(data)} from ARC")
    except Exception as e:
        print(f"[!] ARC load error: {e}")
    return data


def load_openbookqa(sample_size: int = 150, seed: int = 42) -> list[dict]:
    """Loads OpenBookQA dataset."""
    print(f"[*] Loading OpenBookQA ({sample_size} questions)...")
    data = []
    try:
        ds = load_dataset("allenai/openbookqa", "main", split="test")
        sampled = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
        for i, item in enumerate(sampled):
            choices = item["choices"]["text"]
            key = item["answerKey"]
            try:
                idx = item["choices"]["label"].index(key)
                ref_ans = choices[idx]
            except Exception:
                ref_ans = choices[0] if choices else key

            data.append({
                "id": f"obqa_{i:04d}",
                "source_dataset": "OpenBookQA",
                "subject": "science",
                "question": item["question_stem"].strip(),
                "context": "",
                "reference_answer": ref_ans.strip(),
                "options": choices
            })
        print(f"[+] Loaded {len(data)} from OpenBookQA")
    except Exception as e:
        print(f"[!] OpenBookQA load error: {e}")
    return data


def load_race(sample_size: int = 150, seed: int = 42) -> list[dict]:
    """Loads RACE English reading comprehension dataset."""
    print(f"[*] Loading RACE ({sample_size} questions)...")
    data = []
    try:
        ds = load_dataset("EleutherAI/race", split="test")
        all_extracted = []
        for item in ds:
            article = item["article"].strip()
            raw_p = item["problems"]
            problems = ast.literal_eval(raw_p) if isinstance(raw_p, str) else raw_p
            for p in problems:
                opts = p["options"]
                ans_letter = p["answer"].strip()
                ans_idx = "ABCD".index(ans_letter) if ans_letter in "ABCD" else 0
                all_extracted.append({
                    "question": p["question"].strip(),
                    "context": article,
                    "reference_answer": opts[ans_idx].strip(),
                    "options": opts
                })

        random.seed(seed)
        sampled = random.sample(all_extracted, min(sample_size, len(all_extracted)))
        for i, item in enumerate(sampled):
            data.append({
                "id": f"race_{i:04d}",
                "source_dataset": "RACE",
                "subject": "reading_comprehension",
                "question": item["question"],
                "context": item["context"],
                "reference_answer": item["reference_answer"],
                "options": item["options"]
            })
        print(f"[+] Loaded {len(data)} from RACE")
    except Exception as e:
        print(f"[!] RACE load error: {e}")
    return data


def load_squad(sample_size: int = 150, seed: int = 42) -> list[dict]:
    """Loads SQuAD Reading Comprehension dataset."""
    print(f"[*] Loading SQuAD ({sample_size} questions)...")
    data = []
    try:
        ds = load_dataset("rajpurkar/squad", split="validation")
        sampled = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
        for i, item in enumerate(sampled):
            answers = item["answers"]["text"]
            ref_ans = answers[0] if answers else ""
            data.append({
                "id": f"squad_{i:04d}",
                "source_dataset": "SQuAD",
                "subject": "reading_comprehension",
                "question": item["question"].strip(),
                "context": item["context"].strip(),
                "reference_answer": ref_ans.strip(),
                "options": []
            })
        print(f"[+] Loaded {len(data)} from SQuAD")
    except Exception as e:
        print(f"[!] SQuAD load error: {e}")
    return data


def build_focused_benchmark(sample_size: int = 150, seed: int = 42, output_dir: str = "data") -> list[dict]:
    """
    Builds the focused 300-question SciQ x RACE cross-domain benchmark.
    Adds an explicit 'domain' field to each record for downstream analysis.
    """
    sciq_data = []
    race_data = []

    sciq_path = os.path.join(output_dir, "sciq_origin.json")
    race_path = os.path.join(output_dir, "race_reading_origin.json")

    if os.path.exists(sciq_path):
        with open(sciq_path, "r", encoding="utf-8") as f:
            sciq_data = json.load(f)[:sample_size]
        print(f"[+] Loaded {len(sciq_data)} SciQ questions from {sciq_path}")
    else:
        sciq_data = load_sciq(sample_size=sample_size, seed=seed)

    if os.path.exists(race_path):
        with open(race_path, "r", encoding="utf-8") as f:
            race_data = json.load(f)[:sample_size]
        print(f"[+] Loaded {len(race_data)} RACE questions from {race_path}")
    else:
        race_data = load_race(sample_size=sample_size, seed=seed)

    # Stamp explicit cross-domain field
    for item in sciq_data:
        item["domain"] = "Science (SciQ)"
    for item in race_data:
        item["domain"] = "Language/Reading (RACE)"

    focused = sciq_data + race_data
    out_path = os.path.join(output_dir, "sciq_race_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(focused, f, indent=2, ensure_ascii=False)

    print("\n" + "="*75)
    print("FOCUSED CROSS-DOMAIN BENCHMARK GENERATED")
    print("="*75)
    print(f"  SciQ  (Science / Factual Recall)      -> {len(sciq_data)} questions")
    print(f"  RACE  (Language / Reading Comprehension) -> {len(race_data)} questions")
    print(f"  Combined Cross-Domain Total           -> {len(focused)} questions")
    print(f"  Saved to: {out_path}")
    print("="*75)
    return focused


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Preparation for EduBench-Local")
    parser.add_argument("--per-dataset", type=int, default=150,
                        help="Exact number of questions sampled from each dataset (default: 150 each)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for JSON datasets (default: data)")
    parser.add_argument("--focused", action="store_true",
                        help="Build only the focused SciQ x RACE 300-question cross-domain benchmark (uses cached origin files if available)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Focused cross-domain benchmark mode ---
    if args.focused:
        build_focused_benchmark(sample_size=args.per_dataset, seed=args.seed, output_dir=args.output_dir)
        return

    # --- Full 5-dataset pipeline ---
    n = args.per_dataset
    sciq_data = load_sciq(sample_size=n, seed=args.seed)
    arc_data = load_arc(sample_size=n, seed=args.seed)
    obqa_data = load_openbookqa(sample_size=n, seed=args.seed)
    race_data = load_race(sample_size=n, seed=args.seed)
    squad_data = load_squad(sample_size=n, seed=args.seed)

    # 1. Save Individual Origin Files
    origins = {
        "sciq_origin.json": sciq_data,
        "arc_challenge_origin.json": arc_data,
        "openbookqa_origin.json": obqa_data,
        "race_reading_origin.json": race_data,
        "squad_reading_origin.json": squad_data
    }

    for filename, dataset_items in origins.items():
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset_items, f, indent=2, ensure_ascii=False)

    # 2. Save Subject Domain Subsets
    science_data = sciq_data + arc_data + obqa_data
    reading_data = race_data + squad_data
    combined_data = science_data + reading_data

    with open(os.path.join(args.output_dir, "science_domain_origin.json"), "w", encoding="utf-8") as f:
        json.dump(science_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "reading_domain_origin.json"), "w", encoding="utf-8") as f:
        json.dump(reading_data, f, indent=2, ensure_ascii=False)

    # 3. Master Combined File
    with open(os.path.join(args.output_dir, "dataset_sample.json"), "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    # 4. Focused Cross-Domain Benchmark (always build alongside full pipeline)
    build_focused_benchmark(sample_size=n, seed=args.seed, output_dir=args.output_dir)

    print("\n" + "="*75)
    print("DATASET PREPARATION COMPLETED (ORIGIN FILES GENERATED)")
    print("="*75)
    print(f"Origin Files Saved in '{args.output_dir}/':")
    print(f"  1. sciq_origin.json              -> {len(sciq_data)} SciQ Science Exam QA")
    print(f"  2. arc_challenge_origin.json     -> {len(arc_data)} ARC Grade-School Science")
    print(f"  3. openbookqa_origin.json        -> {len(obqa_data)} OpenBookQA Elementary Science")
    print(f"  4. race_reading_origin.json      -> {len(race_data)} RACE Middle/High School English")
    print(f"  5. squad_reading_origin.json     -> {len(squad_data)} SQuAD Reading Comprehension")
    print("-" * 75)
    print(f"  * science_domain_origin.json     -> {len(science_data)} Combined Science Questions")
    print(f"  * reading_domain_origin.json     -> {len(reading_data)} Combined Reading Comp Questions")
    print(f"  * dataset_sample.json            -> {len(combined_data)} Master 750-Question Benchmark")
    print(f"  * sciq_race_benchmark.json       -> {len(sciq_data)+len(race_data)} Focused Cross-Domain Benchmark")
    print("="*75)


if __name__ == "__main__":
    main()
