"""
EduBench-Local: Answer Generator
==================================
Generates answers from Mistral 7B via Ollama with checkpoint/resume support.
Captures metadata: latency, token count, timestamp.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    import ollama
except ImportError:
    print("WARNING: ollama package not installed. Run: pip install ollama")
    ollama = None


def load_checkpoint(checkpoint_path):
    """Load existing answers from checkpoint file."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded checkpoint: {len(data)} answers found")
        return data
    return []


def save_checkpoint(data, checkpoint_path):
    """Save answers to checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_single_answer(prompt, model=None, options=None):
    """
    Generate a single answer from Ollama.
    
    Returns:
        dict with keys: response, latency_sec, token_count, error
    """
    if ollama is None:
        return {
            "response": "",
            "latency_sec": 0,
            "token_count": 0,
            "error": "ollama package not installed",
        }
    
    model = model or config.MODEL_NAME
    options = options or config.GENERATION_OPTIONS
    
    start_time = time.time()
    
    try:
        result = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        
        latency = time.time() - start_time
        response_text = result["message"]["content"].strip()
        
        # Extract token count from response metadata
        token_count = result.get("eval_count", 0)
        if token_count == 0:
            # Rough estimate: ~0.75 tokens per word
            token_count = int(len(response_text.split()) * 1.33)
        
        return {
            "response": response_text,
            "latency_sec": round(latency, 2),
            "token_count": token_count,
            "error": None,
        }
    
    except Exception as e:
        latency = time.time() - start_time
        return {
            "response": "",
            "latency_sec": round(latency, 2),
            "token_count": 0,
            "error": str(e),
        }


def generate_answers(questions, checkpoint_path=None, force=False):
    """
    Generate answers for all questions with checkpoint/resume support.
    
    Args:
        questions: list of standardized question dicts
        checkpoint_path: Path to save/load checkpoints (default: config.RAW_ANSWERS_FILE)
        force: If True, regenerate all answers even if checkpoint exists
        
    Returns:
        list of answer dicts
    """
    from src.prompt_builder import build_prompt
    
    checkpoint_path = Path(checkpoint_path or config.RAW_ANSWERS_FILE)
    
    # Load existing checkpoint
    if not force:
        existing_answers = load_checkpoint(checkpoint_path)
        answered_ids = {a["id"] for a in existing_answers}
    else:
        existing_answers = []
        answered_ids = set()
    
    # Filter to unanswered questions
    remaining = [q for q in questions if q["id"] not in answered_ids]
    
    if not remaining:
        print("  All questions already answered!")
        return existing_answers
    
    print(f"\n{'=' * 60}")
    print(f"Generating answers with {config.MODEL_DISPLAY_NAME}")
    print(f"{'=' * 60}")
    print(f"  Total questions: {len(questions)}")
    print(f"  Already answered: {len(answered_ids)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Checkpoint interval: every {config.CHECKPOINT_INTERVAL} questions")
    print(f"  Checkpoint file: {checkpoint_path}")
    print()
    
    answers = list(existing_answers)
    errors = 0
    
    for i, item in enumerate(tqdm(remaining, desc="Generating answers")):
        prompt = build_prompt(item)
        
        # Try with retries
        result = None
        for attempt in range(config.MAX_RETRIES + 1):
            result = generate_single_answer(prompt)
            if result["error"] is None:
                break
            if attempt < config.MAX_RETRIES:
                print(f"\n  Retry {attempt + 1}/{config.MAX_RETRIES} for {item['id']}: {result['error']}")
                time.sleep(2)
        
        answer_record = {
            "id": item["id"],
            "dataset": item["dataset"],
            "subject": item["subject"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "question_type": item["question_type"],
            "model": config.MODEL_NAME,
            "generated_answer": result["response"],
            "latency_sec": result["latency_sec"],
            "token_count": result["token_count"],
            "error": result["error"],
            "timestamp": datetime.now().isoformat(),
        }
        
        # For MCQ: also store choices and answer_key for later accuracy calc
        if item["question_type"] == "mcq":
            answer_record["choices"] = item.get("choices", [])
            answer_record["answer_key"] = item.get("answer_key", None)
        
        answers.append(answer_record)
        
        if result["error"]:
            errors += 1
        
        # Checkpoint
        if (i + 1) % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(answers, checkpoint_path)
            tqdm.write(f"  💾 Checkpoint saved ({len(answers)} answers)")
    
    # Final save
    save_checkpoint(answers, checkpoint_path)
    
    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Total answers: {len(answers)}")
    print(f"  Errors: {errors}")
    print(f"  Saved to: {checkpoint_path}")
    print(f"{'=' * 60}")
    
    return answers


if __name__ == "__main__":
    from src.dataset_loader import load_dataset_sample
    
    questions = load_dataset_sample()
    
    # Dry run: just 1 question per dataset for testing
    if "--dry-run" in sys.argv:
        seen_datasets = set()
        test_questions = []
        for q in questions:
            if q["dataset"] not in seen_datasets:
                test_questions.append(q)
                seen_datasets.add(q["dataset"])
        questions = test_questions
        print(f"DRY RUN: Testing with {len(questions)} questions (1 per dataset)")
    
    answers = generate_answers(questions)
