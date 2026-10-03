"""
EduBench-Local: Configuration for Orca Mini 3B
=============================================
All project-wide settings for the Orca Mini 3B model evaluation run.
"""

import os
from pathlib import Path

# --- Project Root ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# --- Model Configuration -------------------------------------------------------
MODEL_NAME = "orca-mini:latest"
MODEL_DISPLAY_NAME = "Orca Mini 3B"

# Generation parameters
GENERATION_OPTIONS = {
    "temperature": 0.3,       # Low temperature for more deterministic answers
    "top_p": 0.9,
    "num_predict": 256,       # Max tokens to generate per answer
}

# --- Dataset Configuration ------------------------------------------------------
RANDOM_SEED = 42
SAMPLE_SIZE_PER_DATASET = 50  # 50 questions sampled per dataset (300 total)

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

# --- Paths (Orca-specific results directory) --------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results_orca"
FIGURES_DIR = RESULTS_DIR / "figures"

# Data files
DATASET_SAMPLE_FILE = PROCESSED_DATA_DIR / "dataset_sample_orca.json"

# Results files (Orca-specific)
RAW_ANSWERS_FILE = RESULTS_DIR / "raw_answers.json"
SCORED_RESULTS_FILE = RESULTS_DIR / "scored_results.csv"
LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.csv"

# --- Prompt Templates (same as Mistral for fair comparison) ---------------------
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

# --- LLM Judge Configuration ---------------------------------------------------
JUDGE_MODEL = "orca-mini:latest"  # Self-judging (acknowledged limitation)

JUDGE_PROMPT = """You are a strict automated grading system. 
You MUST output ONLY a single integer from 1 to 5. Do not include any other text, no apologies, and no explanations.

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

SCORE:"""

# --- Runtime Settings -----------------------------------------------------------
OLLAMA_TIMEOUT = 120           # Seconds before timing out a single Ollama call
CHECKPOINT_INTERVAL = 10       # Save checkpoint every N questions
MAX_RETRIES = 2                # Max retries on Ollama failure per question
BERTSCORE_BATCH_SIZE = 32      # Batch size for BERTScore computation

# --- Ensure directories exist ---------------------------------------------------
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
