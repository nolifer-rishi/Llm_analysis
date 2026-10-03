"""
config.py - EduBench-Local Pipeline Configuration
===================================================
Central configuration file. All other pipeline steps import from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).parent.resolve()
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

# Create directories if they don't exist
for _dir in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------
# Number of samples to draw per subject / dataset split (deterministic)
SAMPLE_SIZE_PER_SUBJECT: int = 150

# Random seed for reproducible sampling
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
# List of student models to benchmark (Ollama model tags)
MODELS: list[str] = [
    "qwen2.5:3b",
]

# Model used as the LLM judge for open-ended answer evaluation
JUDGE_MODEL: str = "mistral:7b"

# Ollama base URL (change if Ollama is running on a different host/port)
OLLAMA_BASE_URL: str = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Generation options (passed directly to Ollama)
# ---------------------------------------------------------------------------
GENERATION_OPTIONS: dict = {
    "temperature": 0.2,
    "num_predict": 200,   # max tokens to generate
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE: str = (
    "You are a student answering an exam question. "
    "Answer concisely and accurately.\n\n"
    "Subject: {subject}\n\n"
    "{context_block}"
    "Question: {question}\n\n"
    "Your answer:"
)

JUDGE_PROMPT: str = (
    "You are an expert educational evaluator. Compare the student's answer "
    "to the reference answer and rate it on a scale from 0 to 10.\n\n"
    "Reference answer: {reference_answer}\n"
    "Student's answer: {student_answer}\n\n"
    "Evaluation criteria:\n"
    "- Correctness: Does the student answer convey the same information?\n"
    "- Completeness: Are key facts/concepts present?\n"
    "- Clarity: Is the answer clearly expressed?\n\n"
    'Respond with ONLY a JSON object in this exact format:\n'
    '{{"score": <integer 0-10>, "reasoning": "<one sentence explanation>"}}'
)

# ---------------------------------------------------------------------------
# Output file paths (used by downstream steps)
# ---------------------------------------------------------------------------
DATASET_SAMPLE_PATH = DATA_DIR / "dataset_sample.json"
RESULTS_PATH        = RESULTS_DIR / "results.json"
SUMMARY_PATH        = RESULTS_DIR / "summary.csv"
