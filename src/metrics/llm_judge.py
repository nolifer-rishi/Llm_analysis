"""
EduBench-Local: LLM-as-Judge Metric
=====================================
Uses Mistral 7B (via Ollama) to score generated answers 1-5
for factual correctness and completeness against the reference.

NOTE: This is self-judging (same model generates and judges), which is an
acknowledged limitation documented in the paper.
"""

import json
import time
import sys
import re
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

try:
    import ollama
except ImportError:
    print("WARNING: ollama package not installed. Run: pip install ollama")
    ollama = None


def _parse_judge_score(text):
    """
    Extract an integer 1-5 from the judge's response.
    
    Handles various formats:
    - "3"
    - "Score: 4"
    - "I would rate this a 5."
    - "3/5"
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Try exact single digit first
    if text in ("1", "2", "3", "4", "5"):
        return int(text)
    
    # Look for digits in the text
    digits = re.findall(r'\b([1-5])\b', text)
    if digits:
        return int(digits[0])
    
    # Last resort: any digit 1-5
    for char in text:
        if char in "12345":
            return int(char)
    
    return None


def judge_single(question, reference, generated, model=None):
    """
    Score a single generated answer using the LLM judge.
    
    Args:
        question: str, the question
        reference: str, the reference (correct) answer
        generated: str, the AI-generated answer
        model: str, the judge model (default: config.JUDGE_MODEL)
        
    Returns:
        dict with keys: score (int 1-5 or None), raw_response, latency_sec, error
    """
    if ollama is None:
        return {
            "score": None,
            "raw_response": "",
            "latency_sec": 0,
            "error": "ollama package not installed",
        }
    
    model = model or config.JUDGE_MODEL
    
    prompt = config.JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        generated=generated,
    )
    
    start_time = time.time()
    
    try:
        result = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 128},  # Give Orca room to yapp and output the number
        )
        
        latency = time.time() - start_time
        raw_response = result["message"]["content"].strip()
        score = _parse_judge_score(raw_response)
        
        return {
            "score": score,
            "raw_response": raw_response,
            "latency_sec": round(latency, 2),
            "error": None if score is not None else "Could not parse score",
        }
    
    except Exception as e:
        latency = time.time() - start_time
        return {
            "score": None,
            "raw_response": "",
            "latency_sec": round(latency, 2),
            "error": str(e),
        }


def judge_batch(answers, checkpoint_path=None, force=False):
    """
    Score all answers using the LLM judge with checkpoint/resume.
    
    Args:
        answers: list of answer dicts (must have id, question, reference_answer, generated_answer)
        checkpoint_path: Path for checkpoint file
        force: If True, re-score all answers
        
    Returns:
        dict mapping question id -> judge result dict
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else \
        config.RESULTS_DIR / "judge_checkpoint.json"
    
    # Load checkpoint
    if not force and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Loaded judge checkpoint: {len(existing)} scores found")
    else:
        existing = {}
    
    remaining = [a for a in answers if a["id"] not in existing]
    
    if not remaining:
        print("  All answers already judged!")
        return existing
    
    print(f"\n{'=' * 60}")
    print(f"LLM-as-Judge scoring with {config.JUDGE_MODEL}")
    print(f"{'=' * 60}")
    print(f"  Total answers: {len(answers)}")
    print(f"  Already judged: {len(existing)}")
    print(f"  Remaining: {len(remaining)}")
    print()
    
    results = dict(existing)
    parse_failures = 0
    
    for i, answer in enumerate(tqdm(remaining, desc="Judging answers")):
        # Try with retries
        result = None
        for attempt in range(config.MAX_RETRIES + 1):
            result = judge_single(
                question=answer["question"],
                reference=answer["reference_answer"],
                generated=answer["generated_answer"],
            )
            if result["score"] is not None:
                break
            if attempt < config.MAX_RETRIES:
                time.sleep(1)
        
        results[answer["id"]] = result
        
        if result["score"] is None:
            parse_failures += 1
        
        # Checkpoint
        if (i + 1) % config.CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            tqdm.write(f"  💾 Judge checkpoint saved ({len(results)} scores)")
    
    # Final save
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Judge scoring complete!")
    print(f"  Total scored: {len(results)}")
    print(f"  Parse failures: {parse_failures}")
    
    return results


if __name__ == "__main__":
    # Quick test
    result = judge_single(
        question="What is photosynthesis?",
        reference="Photosynthesis is the process by which plants convert sunlight into energy.",
        generated="Plants use sunlight to produce glucose through photosynthesis.",
    )
    print(f"Score: {result['score']}")
    print(f"Raw response: {result['raw_response']}")
    print(f"Latency: {result['latency_sec']}s")
