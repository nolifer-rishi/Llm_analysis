"""
EduBench-Local: Dataset Loader
===============================
Loads all 5 educational datasets from Hugging Face, samples them,
and standardizes into a unified schema.
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _extract_choice_text(choices_data, answer_key, dataset_name):
    """Extract the text of the correct answer from MCQ choices."""
    if dataset_name in ("arc_easy", "arc_challenge"):
        # ARC: choices = {"text": [...], "label": [...]}
        labels = choices_data["label"]
        texts = choices_data["text"]
        if answer_key in labels:
            idx = labels.index(answer_key)
            return texts[idx]
        return texts[0] if texts else ""
    elif dataset_name == "openbookqa":
        # OpenBookQA: same structure as ARC
        labels = choices_data["label"]
        texts = choices_data["text"]
        if answer_key in labels:
            idx = labels.index(answer_key)
            return texts[idx]
        return texts[0] if texts else ""
    return ""


def _format_choices(choices_data, dataset_name):
    """Format choices as a list of (label, text) tuples."""
    if dataset_name in ("arc_easy", "arc_challenge", "openbookqa"):
        labels = choices_data["label"]
        texts = choices_data["text"]
        return list(zip(labels, texts))
    return []


def _standardize_sciq(example, idx):
    """Standardize a SciQ example."""
    return {
        "id": f"sciq_{idx:04d}",
        "dataset": "SciQ",
        "subject": config.DATASETS["sciq"]["subject"],
        "question_type": "open_ended",
        "question": example["question"],
        "context": example.get("support", ""),
        "reference_answer": example["correct_answer"],
        "choices": [],
        "answer_key": None,
    }


def _standardize_openbookqa(example, idx):
    """Standardize an OpenBookQA example."""
    choices = _format_choices(example["choices"], "openbookqa")
    answer_key = example["answerKey"]
    ref_answer = _extract_choice_text(example["choices"], answer_key, "openbookqa")
    
    return {
        "id": f"openbookqa_{idx:04d}",
        "dataset": "OpenBookQA",
        "subject": config.DATASETS["openbookqa"]["subject"],
        "question_type": "mcq",
        "question": example["question_stem"],
        "context": example.get("fact1", ""),
        "reference_answer": ref_answer,
        "choices": [{"label": l, "text": t} for l, t in choices],
        "answer_key": answer_key,
    }


def _standardize_arc(example, idx, variant="easy"):
    """Standardize an ARC example (easy or challenge)."""
    key = f"arc_{variant}"
    prefix = f"arc_{variant}"
    choices = _format_choices(example["choices"], key)
    answer_key = example["answerKey"]
    ref_answer = _extract_choice_text(example["choices"], answer_key, key)
    
    return {
        "id": f"{prefix}_{idx:04d}",
        "dataset": f"ARC-{'Easy' if variant == 'easy' else 'Challenge'}",
        "subject": config.DATASETS[key]["subject"],
        "question_type": "mcq",
        "question": example["question"],
        "context": "",
        "reference_answer": ref_answer,
        "choices": [{"label": l, "text": t} for l, t in choices],
        "answer_key": answer_key,
    }


def _standardize_race(example, idx):
    """Standardize a RACE example."""
    options = example["options"]
    answer_letter = example["answer"]  # A, B, C, or D
    answer_idx = ord(answer_letter) - ord("A")
    ref_answer = options[answer_idx] if answer_idx < len(options) else options[0]
    
    labels = ["A", "B", "C", "D"]
    choices = list(zip(labels[:len(options)], options))
    
    return {
        "id": f"race_{idx:04d}",
        "dataset": "RACE",
        "subject": config.DATASETS["race"]["subject"],
        "question_type": "mcq",
        "question": example["question"],
        "context": example.get("article", ""),
        "reference_answer": ref_answer,
        "choices": [{"label": l, "text": t} for l, t in choices],
        "answer_key": answer_letter,
    }


def _standardize_squad(example, idx):
    """Standardize a SQuAD example."""
    answers = example["answers"]
    ref_answer = answers["text"][0] if answers["text"] else ""
    
    return {
        "id": f"squad_{idx:04d}",
        "dataset": "SQuAD",
        "subject": config.DATASETS["squad"]["subject"],
        "question_type": "open_ended",
        "question": example["question"],
        "context": example.get("context", ""),
        "reference_answer": ref_answer,
        "choices": [],
        "answer_key": None,
    }


def load_and_standardize_dataset(dataset_key):
    """Load a single dataset from HuggingFace and standardize it."""
    ds_config = config.DATASETS[dataset_key]
    hf_id = ds_config["hf_identifier"]
    split = ds_config["split"]
    hf_config = ds_config.get("hf_config", None)
    
    print(f"  Loading {dataset_key} ({hf_id})...")
    
    if hf_config:
        ds = load_dataset(hf_id, hf_config, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    
    # Shuffle and sample
    ds = ds.shuffle(seed=config.RANDOM_SEED)
    sample_size = min(config.SAMPLE_SIZE_PER_DATASET, len(ds))
    ds = ds.select(range(sample_size))
    
    print(f"  Sampled {sample_size} questions from {len(ds)} available")
    
    # Standardize
    standardized = []
    for idx, example in enumerate(ds):
        if dataset_key == "sciq":
            item = _standardize_sciq(example, idx)
        elif dataset_key == "openbookqa":
            item = _standardize_openbookqa(example, idx)
        elif dataset_key == "arc_easy":
            item = _standardize_arc(example, idx, variant="easy")
        elif dataset_key == "arc_challenge":
            item = _standardize_arc(example, idx, variant="challenge")
        elif dataset_key == "race":
            item = _standardize_race(example, idx)
        elif dataset_key == "squad":
            item = _standardize_squad(example, idx)
        else:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        standardized.append(item)
    
    return standardized


def load_all_datasets():
    """Load and standardize all configured datasets."""
    all_data = []
    
    print("=" * 60)
    print("Loading and standardizing all datasets...")
    print("=" * 60)
    
    for dataset_key in config.DATASETS:
        print(f"\n[{dataset_key}]")
        items = load_and_standardize_dataset(dataset_key)
        all_data.extend(items)
        print(f"  ✓ {len(items)} questions standardized")
    
    print(f"\n{'=' * 60}")
    print(f"Total questions: {len(all_data)}")
    print(f"{'=' * 60}")
    
    return all_data


def save_dataset_sample(data=None):
    """Load all datasets and save as a single JSON file."""
    if data is None:
        data = load_all_datasets()
    
    output_path = config.DATASET_SAMPLE_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(data)} questions to {output_path}")
    
    # Print summary
    from collections import Counter
    dataset_counts = Counter(item["dataset"] for item in data)
    print("\nDataset breakdown:")
    for ds, count in sorted(dataset_counts.items()):
        print(f"  {ds}: {count} questions")
    
    return data


def load_dataset_sample():
    """Load the previously saved dataset sample."""
    path = config.DATASET_SAMPLE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset sample not found at {path}. "
            "Run save_dataset_sample() first."
        )
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} questions from {path}")
    return data


if __name__ == "__main__":
    save_dataset_sample()
