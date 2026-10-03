"""
EduBench-Local: Central Configuration
=====================================
All project-wide settings in one place.
"""

import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ─── Model Configuration ────────────────────────────────────────────────────
MODEL_NAME = "mistral:7b"
MODEL_DISPLAY_NAME = "Mistral 7B"

# Generation parameters
GENERATION_OPTIONS = {
    "temperature": 0.3,       # Low temperature for more deterministic answers
    "top_p": 0.9,
    "num_predict": 256,       # Max tokens to generate per answer
}

# ─── Dataset Configuration ──────────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE_PER_DATASET = 100  # Number of questions sampled per dataset

DATASETS = {
    "sciq": {
        "hf_identifier": "allenai/sciq",
        "split": "test",
        "subject": "Science",
        "type": "open_ended",
        "description": "Science exam questions with ground-truth answers and supporting evidence",
    },
    "openbookqa": {
        "hf_identifier": "allenai/openbookqa",
        "split": "test",
        "subject": "Science (Reasoning)",
        "type": "mcq",
        "description": "Elementary science reasoning, multiple-choice",
    },
    "arc_easy": {
        "hf_identifier": "allenai/ai2_arc",
        "hf_config": "ARC-Easy",
        "split": "test",
        "subject": "Science (Grade School)",
        "type": "mcq",
        "description": "Grade-school science questions (easy subset)",
    },
    "arc_challenge": {
        "hf_identifier": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "split": "test",
        "subject": "Science (Grade School - Hard)",
        "type": "mcq",
        "description": "Grade-school science questions (challenge subset)",
    },
    "race": {
        "hf_identifier": "ehovy/race",
        "hf_config": "middle",
        "split": "test",
        "subject": "Reading Comprehension",
        "type": "mcq",
        "description": "English reading comprehension (middle school exams)",
    },
    "squad": {
        "hf_identifier": "rajpurkar/squad",
        "split": "validation",  # SQuAD test set has no public answers
        "subject": "General Knowledge (Extractive QA)",
        "type": "open_ended",
        "description": "General reading comprehension QA (extractive)",
    },
}

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Data files
DATASET_SAMPLE_FILE = PROCESSED_DATA_DIR / "dataset_sample.json"

# Results files
RAW_ANSWERS_FILE = RESULTS_DIR / "raw_answers.json"
SCORED_RESULTS_FILE = RESULTS_DIR / "scored_results.csv"
LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.csv"

# ─── Prompt Templates ───────────────────────────────────────────────────────
PROMPT_OPEN_ENDED = """You are a helpful and knowledgeable tutor.
Answer the following question clearly and concisely, in 2-4 sentences.

Question: {question}
{context_block}
Answer:"""

PROMPT_MCQ = """You are a helpful and knowledgeable tutor.
Answer the following multiple-choice question. First give the letter of the correct answer, then explain your reasoning in 1-2 sentences.

Question: {question}
{context_block}Choices:
{choices_block}

Answer:"""

# ─── LLM Judge Configuration ────────────────────────────────────────────────
JUDGE_MODEL = "mistral:7b"  # Self-judging (acknowledged limitation)

JUDGE_PROMPT = """You are grading a student-facing AI tutor's answer.

Question: {question}
Reference (correct) answer: {reference}
AI-generated answer: {generated}

Rate the AI-generated answer from 1 to 5 for factual correctness and completeness relative to the reference answer.

Scale:
1 = Completely wrong or irrelevant
2 = Mostly wrong, minor correct elements
3 = Partially correct but incomplete or contains errors
4 = Mostly correct with minor issues
5 = Fully correct and complete

Respond with ONLY a single integer from 1 to 5."""

# ─── Runtime Settings ────────────────────────────────────────────────────────
OLLAMA_TIMEOUT = 120           # Seconds before timing out a single Ollama call
CHECKPOINT_INTERVAL = 10       # Save checkpoint every N questions
MAX_RETRIES = 2                # Max retries on Ollama failure per question
BERTSCORE_BATCH_SIZE = 32      # Batch size for BERTScore computation

# ─── Ensure directories exist ────────────────────────────────────────────────
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
