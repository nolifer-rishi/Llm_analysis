# -*- coding: utf-8 -*-
"""
gemini_evaluate.py
------------------
Evaluates a micro-batch of 1 or 2 questions per dataset (5-10 questions total)
using the official Google GenAI SDK.

Primary Model:
  gemini-2.5-flash (with automatic fallback to gemini-3.8-flash / gemini-3.6-flash
  if Google's API reports 404 'no longer available to new users' or daily quota limits).

Datasets:
  SciQ              <- subject: science
  OpenBookQA        <- subject: general_science
  ARC-Challenge     <- subject: science_challenge
  RACE              <- subject: reading_comprehension
  SQuAD v1.1        <- subject: reading_comprehension_squad

Settings:
  Micro-sample : 2 questions per dataset (10 questions total; configurable via --samples)
  Temperature  : 0.0 (deterministic)
  Delay        : 4 seconds between API calls (time.sleep(4))

Output: results/gemini_raw_answers.json (reset fresh on start, updated incrementally
        as formatted JSON after every single question).
"""

import sys
import io

# Force line-buffered UTF-8 stdout (real-time log output on Windows)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

from google import genai
from google.genai import types
from google.genai.models import Models

# Suppress the verbose SDK AFC warning from polluting stdout
Models._logged_afc_warning = True

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_ANSWERS_PATH = os.path.join("results", "raw_answers.json")
OUTPUT_PATH      = os.path.join("results", "gemini_raw_answers.json")

# Primary model requested is gemini-2.5-flash.
# Google API now advises new keys: "models/gemini-2.5-flash is no longer available to new users.
# Please update your code to use models/gemini-3.8-flash".
# We include gemini-3.8-flash and gemini-3.6-flash as automatic fallbacks.
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

DEFAULT_SAMPLES_PER_DATASET = 2   # 2 per dataset × 5 datasets = 10 questions max
TEMPERATURE                 = 0.0 # deterministic output
MAX_OUTPUT_TOKENS           = 256
INTER_REQUEST_SLEEP         = 4   # seconds between calls (time.sleep(4))
MAX_RETRIES                 = 4   # retries for transient 429-PerMinute / 503

SUBJECT_LABEL = {
    "science":                     "SciQ",
    "general_science":             "OpenBookQA",
    "science_challenge":           "ARC-Challenge",
    "reading_comprehension":       "RACE",
    "reading_comprehension_squad": "SQuAD v1.1",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError(
            "API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "as an environment variable."
        )
    return key


def build_prompt(record: dict) -> str:
    context  = (record.get("context")  or "").strip()
    question = (record.get("question") or "").strip()
    if context:
        return (
            "Read the passage carefully, then answer in as few words as possible.\n\n"
            f"Passage:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    return (
        "Answer in as few words as possible.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def parse_retry_delay(err: str, default: float = 20.0) -> float:
    m = re.search(r"retry[Dd]elay['\"]:\s*['\"]?(\d+(?:\.\d+)?)", err)
    if m:
        return float(m.group(1)) + 3.0
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 3.0
    return default


def is_per_minute_quota(err: str) -> bool:
    return "PerMinute" in err or "per_minute" in err.lower()


def is_per_day_quota(err: str) -> bool:
    return "PerDay" in err or "per_day" in err.lower()


def is_not_found_or_discontinued(err: str) -> bool:
    err_low = err.lower()
    return ("404" in err or "not_found" in err_low or
            "no longer available" in err_low or "not supported" in err_low)


def load_micro_sample(samples_per_dataset: int) -> list:
    """Load raw_answers.json and return exactly samples_per_dataset records
    per subject, deduplicated by question id."""
    print(f"[Load]  Reading '{RAW_ANSWERS_PATH}' ...")
    with open(RAW_ANSWERS_PATH, encoding="utf-8") as fh:
        all_records = json.load(fh)

    seen: set = set()
    unique: list = []
    for rec in all_records:
        qid = rec.get("id", "")
        if qid not in seen:
            seen.add(qid)
            unique.append(rec)

    by_subject: dict = defaultdict(list)
    for rec in unique:
        by_subject[rec.get("subject", "unknown")].append(rec)

    sample: list = []
    print(f"\n{'─'*55}")
    print(f"  {'Dataset':<30} {'Available':>9}  {'Selected':>8}")
    print(f"{'─'*55}")
    for subj in sorted(SUBJECT_LABEL.keys()):
        recs = by_subject.get(subj, [])
        chosen = recs[:samples_per_dataset]
        sample.extend(chosen)
        label = SUBJECT_LABEL.get(subj, subj)
        print(f"  {label:<30} {len(recs):>9}  {len(chosen):>8}")
    print(f"{'─'*55}")
    print(f"  {'TOTAL MICRO-TARGET':<30} {'':>9}  {len(sample):>8}")
    print(f"{'─'*55}\n")
    return sample


def extract_answer_text(resp) -> str:
    """Safely extract generated text from response object without crashing on None."""
    if resp is None:
        return ""
    if getattr(resp, "text", None):
        return resp.text.strip()
    # Check candidates structure if text property is empty
    if hasattr(resp, "candidates") and resp.candidates:
        candidate = resp.candidates[0]
        if hasattr(candidate, "content") and candidate.content:
            parts = getattr(candidate.content, "parts", []) or []
            combined = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
            if combined:
                return combined.strip()
    return ""


# ── API call with retry / model rotation ──────────────────────────────────────

def call_with_retry(client, prompt: str, exhausted: set) -> tuple:
    """Returns (answer, error_msg, model_used)."""
    for attempt in range(MAX_RETRIES + 1):
        model = next((m for m in MODEL_CANDIDATES if m not in exhausted), None)
        if model is None:
            return None, "All models exhausted their quota or are unavailable.", "none"
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            answer = extract_answer_text(resp)
            return answer, None, model

        except Exception as exc:
            err = str(exc)

            # 404 / Discontinued: switch model immediately and do not reuse this model
            if is_not_found_or_discontinued(err):
                print(f"  [MODEL-404]  {model} unavailable for this API key — falling back to next model.")
                exhausted.add(model)
                continue

            # 429 Daily Limit: switch model immediately
            if "429" in err and is_per_day_quota(err):
                print(f"  [QUOTA-DAY]  {model} daily limit reached — switching model.")
                exhausted.add(model)
                continue

            # 429 Rate Limit (per minute): sleep then retry
            if "429" in err and is_per_minute_quota(err):
                wait = parse_retry_delay(err)
                print(f"  [QUOTA-MIN]  Rate limited on {model} — sleeping {wait:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES}) ...")
                time.sleep(wait)
                continue

            # 503 Service Unavailable / High Demand: retry or rotate if stuck
            if "503" in err or "UNAVAILABLE" in err:
                wait = 6 + attempt * 4
                print(f"  [503-WAIT]   {model} high demand — sleeping {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES}) ...")
                time.sleep(wait)
                if attempt >= 2:
                    print(f"  [503-FALLBACK] {model} consistently busy — trying fallback model.")
                    exhausted.add(model)
                continue

            return None, err, model        # non-retriable error

    return None, f"Max retries ({MAX_RETRIES}) exceeded.", \
           next((m for m in MODEL_CANDIDATES if m not in exhausted), "none")


def _save_incremental(path: str, records: list):
    """Write records atomically as clean, formatted JSON so it can be inspected immediately."""
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(samples_per_dataset: int = DEFAULT_SAMPLES_PER_DATASET):
    api_key = get_api_key()
    client  = genai.Client(api_key=api_key)

    exhausted: set = set()

    print()
    print("=" * 60)
    print("  EduBench-Local  |  Gemini Micro-Batch Evaluation")
    print("=" * 60)
    print(f"  Primary model  : {MODEL_CANDIDATES[0]}")
    print(f"  Fallback models: {MODEL_CANDIDATES[1:]}")
    print(f"  Micro-sample   : {samples_per_dataset} per dataset × 5 datasets = "
          f"{samples_per_dataset * 5} total")
    print(f"  Delay          : {INTER_REQUEST_SLEEP}s between calls (time.sleep({INTER_REQUEST_SLEEP}))")
    print(f"  Temperature    : {TEMPERATURE}  |  Max tokens: {MAX_OUTPUT_TOKENS}")
    print("=" * 60)

    records = load_micro_sample(samples_per_dataset)

    # ── Hard-delete previous output file so the evaluation starts 100% fresh ───
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"[Reset] Deleted previous '{OUTPUT_PATH}'.")
    _save_incremental(OUTPUT_PATH, [])
    print(f"[Reset] '{OUTPUT_PATH}' created fresh — starting micro-generation.\n")

    saved_records = []
    answered = 0
    errors   = 0
    current_subject = None

    for idx, rec in enumerate(records, start=1):
        subject = rec.get("subject", "unknown")
        label   = SUBJECT_LABEL.get(subject, subject)

        if subject != current_subject:
            if current_subject is not None:
                print()
            current_subject = subject
            print(f"\n  [{label}]")
            print(f"  {'-'*50}")

        # Skip immediately if all models exhausted
        if all(m in exhausted for m in MODEL_CANDIDATES):
            print(f"  [SKIP] {idx:>2}/{len(records)}  — all models exhausted")
            out_rec = {
                "id": rec.get("id", f"q_{idx}"),
                "source_dataset": label,
                "subject": subject,
                "question": rec.get("question", ""),
                "context": rec.get("context", ""),
                "prompt": "",
                "reference_answer": rec.get("reference_answer", ""),
                "gemini_answer": None,
                "model_used": "none",
                "error": "All models exhausted their quota or are unavailable.",
            }
            saved_records.append(out_rec)
            _save_incremental(OUTPUT_PATH, saved_records)
            errors += 1
            continue

        prompt_text = build_prompt(rec)
        answer, err_msg, model_used = call_with_retry(client, prompt_text, exhausted)

        status  = "OK " if err_msg is None else "ERR"
        preview = repr((answer or err_msg or "")[:55])
        print(f"  [{status}] {idx:>2}/{len(records)}  "
              f"model={model_used:<22}  {rec.get('id','?')}")
        print(f"         ans={preview}")

        out_rec = {
            "id": rec.get("id", f"q_{idx}"),
            "source_dataset": label,
            "subject": subject,
            "question": rec.get("question", ""),
            "context": rec.get("context", ""),
            "prompt": prompt_text,
            "reference_answer": rec.get("reference_answer", ""),
            "gemini_answer": answer,
            "model_used": model_used,
            "error": err_msg,
        }
        saved_records.append(out_rec)
        _save_incremental(OUTPUT_PATH, saved_records)

        if err_msg is None:
            answered += 1
        else:
            errors += 1

        # Delay between calls
        if idx < len(records):
            time.sleep(INTER_REQUEST_SLEEP)

    _print_summary(records, answered, errors)


def _print_summary(records, answered, errors):
    data = []
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            pass

    by_ds  = Counter(r["source_dataset"] for r in data if r.get("gemini_answer"))
    by_mod = Counter(r.get("model_used", "?") for r in data if r.get("gemini_answer"))

    print()
    print("=" * 60)
    print("  MICRO-BATCH EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions targeted : {len(records)}")
    print(f"  Successfully answered    : {answered}")
    print(f"  Errors / Skipped         : {errors}")
    print()
    print("  Answered by dataset:")
    for ds in sorted(SUBJECT_LABEL.values()):
        cnt = by_ds.get(ds, 0)
        print(f"    {ds:<28} {cnt}")
    print()
    print("  Answered by model:")
    if by_mod:
        for mod, cnt in sorted(by_mod.items()):
            print(f"    {mod:<28} {cnt}")
    else:
        print("    (None)")
    print()
    print(f"  Output File : {os.path.abspath(OUTPUT_PATH)}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gemini on a micro-batch of benchmark questions.")
    parser.add_argument(
        "--samples",
        type=int,
        default=int(os.environ.get("SAMPLES_PER_DATASET", DEFAULT_SAMPLES_PER_DATASET)),
        help="Number of questions per dataset (e.g. 1 or 2, default: 2 -> 10 total)",
    )
    args = parser.parse_args()
    run_evaluation(samples_per_dataset=args.samples)
