"""
step2_generate_answers.py - EduBench-Local Pipeline Step 2
===========================================================
For every model in config.MODELS, sends each question from
data/dataset_sample.json to Ollama and records the response.

Features:
  - Resumable: skips (model, question_id) pairs already in raw_answers.json
  - Incremental saves: writes after every answer (crash-safe)
  - tqdm progress bars: one bar per model
  - Tracks latency (seconds) and token counts per response

Output schema per record in results/raw_answers.json:
  {
    "model":            str,
    "id":               str,
    "subject":          str,
    "question":         str,
    "context":          str,
    "reference_answer": str,
    "student_answer":   str,
    "latency_s":        float,
    "prompt_tokens":    int | null,
    "completion_tokens":int | null,
    "total_tokens":     int | null,
    "error":            str | null   # set if the Ollama call failed
  }

Usage:
  python step2_generate_answers.py
"""

import json
import sys
import time
from pathlib import Path

import ollama
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_ANSWERS_PATH = config.RESULTS_DIR / "raw_answers.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not config.DATASET_SAMPLE_PATH.exists():
        print(
            f"ERROR: Dataset not found at {config.DATASET_SAMPLE_PATH}\n"
            "       Run step1_prepare_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(config.DATASET_SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_existing_results() -> list[dict]:
    if RAW_ANSWERS_PATH.exists():
        with open(RAW_ANSWERS_PATH, encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("WARNING: raw_answers.json is corrupt, starting fresh.")
                return []
    return []


def save_results(results: list[dict]) -> None:
    RAW_ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def build_done_set(results: list[dict]) -> set[tuple[str, str]]:
    """Return set of (model, question_id) pairs already completed."""
    return {(r["model"], r["id"]) for r in results}


def build_prompt(item: dict) -> str:
    context_block = (
        f"Context:\n{item['context']}\n\n"
        if item.get("context")
        else ""
    )
    return config.PROMPT_TEMPLATE.format(
        subject=item["subject"],
        context_block=context_block,
        question=item["question"],
    )


def call_ollama(model: str, prompt: str) -> dict:
    """
    Call ollama.chat() and return a dict with:
      answer, latency_s, prompt_tokens, completion_tokens, total_tokens, error
    """
    t0 = time.perf_counter()
    error = None
    answer = ""
    prompt_tokens = completion_tokens = total_tokens = None

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=config.GENERATION_OPTIONS,
        )
        # .content can be None for some Ollama builds; guard against that
        raw_content = response["message"]["content"]
        answer = raw_content.strip() if raw_content else ""

        # Token usage: Ollama stores counts at the top level of ChatResponse
        prompt_tokens     = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        total_tokens      = (
            (prompt_tokens or 0) + (completion_tokens or 0)
        ) or None

    except Exception as exc:
        error = str(exc)

    latency_s = time.perf_counter() - t0

    return {
        "answer":            answer,
        "latency_s":         round(latency_s, 3),
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "error":             error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(model: str, dataset: list[dict], results: list[dict]) -> None:
    done = build_done_set(results)
    pending = [item for item in dataset if (model, item["id"]) not in done]

    if not pending:
        print(f"  [{model}] All {len(dataset)} questions already answered. Skipping.")
        return

    print(f"  [{model}] {len(done)} already done, {len(pending)} remaining.")

    errors = 0
    pbar = tqdm(
        pending,
        desc=f"  {model}",
        unit="q",
        dynamic_ncols=True,
        colour="cyan",
    )

    for item in pbar:
        prompt = build_prompt(item)
        call   = call_ollama(model, prompt)

        record = {
            "model":             model,
            "id":                item["id"],
            "subject":           item["subject"],
            "question":          item["question"],
            "context":           item.get("context", ""),
            "reference_answer":  item["reference_answer"],
            "student_answer":    call["answer"],
            "latency_s":         call["latency_s"],
            "prompt_tokens":     call["prompt_tokens"],
            "completion_tokens": call["completion_tokens"],
            "total_tokens":      call["total_tokens"],
            "error":             call["error"],
        }
        results.append(record)

        if call["error"]:
            errors += 1
            pbar.set_postfix({"errors": errors, "last_err": call["error"][:40]})
        else:
            tok  = call["completion_tokens"] or "?"
            lat  = call["latency_s"]
            pbar.set_postfix({"latency": f"{lat:.1f}s", "tokens": tok})

        # Incremental save after every answer
        save_results(results)

    pbar.close()
    answered = len(pending) - errors
    print(
        f"  [{model}] Done — {answered}/{len(pending)} answered, "
        f"{errors} error(s). "
        f"Avg latency: "
        + (
            f"{sum(r['latency_s'] for r in results if r['model']==model) / len([r for r in results if r['model']==model]):.2f}s"
            if results else "N/A"
        )
    )


def main() -> None:
    print("=" * 60)
    print("EduBench-Local  |  Step 2: Generate Answers")
    print("=" * 60)
    print(f"  Models    : {config.MODELS}")
    print(f"  Options   : {config.GENERATION_OPTIONS}")
    print(f"  Output    : {RAW_ANSWERS_PATH}")
    print()

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} questions from {config.DATASET_SAMPLE_PATH.name}")

    results = load_existing_results()
    print(f"Loaded {len(results)} existing results (resuming if any)\n")

    for model in config.MODELS:
        print(f"\n--- Model: {model} ---")

        # Verify Ollama can reach the model before starting
        try:
            ollama.show(model)
        except Exception as e:
            err = str(e)
            if "not found" in err.lower() or "pull" in err.lower():
                print(f"  WARNING: Model '{model}' not found locally.")
                print(f"  Run:  ollama pull {model}")
                print(f"  Skipping this model.\n")
                continue
            # Other errors (connection refused, etc.)
            print(f"  ERROR connecting to Ollama: {err}")
            print(f"  Make sure Ollama is running:  ollama serve")
            print(f"  Skipping this model.\n")
            continue

        run_model(model, dataset, results)

    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for model in config.MODELS:
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            print(f"  {model}: no results")
            continue
        errors  = sum(1 for r in model_results if r["error"])
        lats    = [r["latency_s"] for r in model_results]
        avg_lat = sum(lats) / len(lats) if lats else 0
        print(
            f"  {model}: {len(model_results)} answers, "
            f"{errors} errors, avg latency {avg_lat:.2f}s"
        )
    print(f"\nResults saved to: {RAW_ANSWERS_PATH}")
    print("Step 2 complete.")


if __name__ == "__main__":
    main()
