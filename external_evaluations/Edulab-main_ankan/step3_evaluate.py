"""
step3_evaluate.py - EduBench-Local Pipeline Step 3
====================================================
Evaluates model answers in results/raw_answers.json using four metrics:

  1. Exact Match (EM)
       Normalised string comparison (lowercase, strip punctuation/whitespace).
       Binary 0/1 per answer.

  2. ROUGE-L
       Longest common subsequence F1 via the rouge_score library.
       Range: [0.0, 1.0].

  3. BERTScore (F1)
       Contextual embedding similarity via the bert_score library.
       Computed in one batched pass per model (fast & memory-efficient).
       Range: [0.0, 1.0] (typically 0.8–1.0 for English).

  4. LLM-as-Judge score (1-5)
       Uses config.JUDGE_MODEL (mistral:7b) with config.JUDGE_PROMPT.
       The judge returns a JSON {score: 0-10, reasoning: "..."}.
       We normalise that to 1-5 for the output CSVs.

Features:
  - Resumable: skips (model, id) pairs already in scored_results.csv
  - Incremental CSV writes (crash-safe): appends rows as they are scored
  - BERTScore batched over all records after LLM-judge loop finishes
  - tqdm progress bars with live score display
  - No pandas/numpy required (pure stdlib csv module)

Output files:
  results/scored_results.csv    - one row per (model, question)
  results/leaderboard.csv       - one row per model (overall aggregated)
  results/subject_leaderboard.csv - one row per (model, subject)

Usage:
  python step3_evaluate.py
"""

import csv
import json
import re
import string
import sys
from pathlib import Path

import ollama
from rouge_score import rouge_scorer as rs_lib
from bert_score import score as bert_score_fn
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ANSWERS_PATH         = config.RESULTS_DIR / "raw_answers.json"
SCORED_RESULTS_PATH      = config.RESULTS_DIR / "scored_results.csv"
LEADERBOARD_PATH         = config.RESULTS_DIR / "leaderboard.csv"
SUBJECT_LEADERBOARD_PATH = config.RESULTS_DIR / "subject_leaderboard.csv"

# Only score answers produced by this model; set to None to score all models.
TARGET_MODEL: str | None = "qwen2.5:3b"

SCORED_FIELDNAMES = [
    "model", "id", "subject",
    "question", "reference_answer", "student_answer",
    "exact_match",
    "rouge_l",
    "bert_score_f1",
    "llm_score_0_10", "llm_score_1_5", "llm_reasoning",
    "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
    "judge_error",
]

LEADERBOARD_FIELDNAMES = [
    "rank", "model", "n_questions",
    "exact_match_rate",
    "avg_rouge_l",
    "avg_bert_score_f1",
    "avg_llm_score_1_5",
    "avg_llm_score_0_10",
    "avg_latency_s",
    "n_errors",
    "subjects",
]

SUBJECT_LEADERBOARD_FIELDNAMES = [
    "rank", "model", "subject", "n_questions",
    "exact_match_rate",
    "avg_rouge_l",
    "avg_bert_score_f1",
    "avg_llm_score_1_5",
    "avg_llm_score_0_10",
]


# ---------------------------------------------------------------------------
# Exact-match helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(reference: str, candidate: str) -> int:
    """Return 1 if normalised strings match, else 0."""
    return int(_normalise(reference) == _normalise(candidate))


# ---------------------------------------------------------------------------
# ROUGE-L helper
# ---------------------------------------------------------------------------

_ROUGE_SCORER = rs_lib.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(reference: str, candidate: str) -> float:
    """Return ROUGE-L F1 score in [0, 1]."""
    result = _ROUGE_SCORER.score(reference, candidate)
    return round(result["rougeL"].fmeasure, 6)


# ---------------------------------------------------------------------------
# LLM judge helpers
# ---------------------------------------------------------------------------

def _call_judge(reference_answer: str, student_answer: str) -> dict:
    """
    Ask JUDGE_MODEL to score the student answer using JUDGE_PROMPT.
    Returns:
      {"score_0_10": int, "score_1_5": int, "reasoning": str, "error": str|None}
    """
    prompt = config.JUDGE_PROMPT.format(
        reference_answer=reference_answer,
        student_answer=student_answer,
    )
    error = None
    score_0_10 = None
    reasoning  = ""

    try:
        response = ollama.chat(
            model=config.JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 150},
        )
        raw = response["message"]["content"]
        raw = raw.strip() if raw else ""

        # Extract the JSON object even if the model wraps it in prose
        match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', raw)
        if match:
            parsed     = json.loads(match.group())
            score_0_10 = int(parsed.get("score", 0))
            reasoning  = str(parsed.get("reasoning", "")).strip()
        else:
            # Fallback: scan for the first bare integer 0-10
            nums = re.findall(r'\b(\d+)\b', raw)
            score_0_10 = int(nums[0]) if nums else 0
            reasoning  = raw[:200]

        # Clamp to [0, 10]
        score_0_10 = max(0, min(10, score_0_10))

    except Exception as exc:
        error      = str(exc)
        score_0_10 = 0

    # Normalise 0-10 -> 1-5  (0-1->1, 2-3->2, 4-5->3, 6-7->4, 8-10->5)
    score_1_5 = max(1, round((score_0_10 / 10) * 4) + 1) if score_0_10 is not None else 1

    return {
        "score_0_10": score_0_10,
        "score_1_5":  score_1_5,
        "reasoning":  reasoning,
        "error":      error,
    }


# ---------------------------------------------------------------------------
# BERTScore batch helper
# ---------------------------------------------------------------------------

def compute_bert_scores(references: list[str], candidates: list[str],
                        lang: str = "en") -> list[float]:
    """
    Compute BERTScore F1 for parallel reference/candidate lists.
    Returns a list of float values in [0, 1].
    """
    if not references:
        return []
    print(f"\n  Computing BERTScore for {len(references):,} pairs "
          f"(this may take a minute) ...")
    _, _, F = bert_score_fn(
        candidates, references,
        lang=lang,
        verbose=False,
        device=None,   # auto-select CPU / CUDA
    )
    return [round(float(v), 6) for v in F.tolist()]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_existing_scored() -> set[tuple[str, str]]:
    """Return set of (model, id) pairs already in scored_results.csv."""
    done = set()
    if SCORED_RESULTS_PATH.exists():
        with open(SCORED_RESULTS_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["model"], row["id"]))
    return done


def open_scored_csv(append: bool):
    """Open scored_results.csv for writing (header) or appending."""
    mode = "a" if append else "w"
    f = open(SCORED_RESULTS_PATH, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=SCORED_FIELDNAMES)
    if not append:
        writer.writeheader()
    return f, writer


def _read_scored_rows() -> list[dict]:
    """Read all rows from scored_results.csv.

    Returns an empty list when the file does not exist, is empty, or contains
    only a header line with no data rows.
    """
    if not SCORED_RESULTS_PATH.exists():
        return []
    if SCORED_RESULTS_PATH.stat().st_size == 0:
        return []
    with open(SCORED_RESULTS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows




# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def write_leaderboard(rows: list[dict]) -> list[dict]:
    """Compute per-model overall aggregates from in-memory rows.

    Writes leaderboard.csv and returns the ranked list of row dicts so the
    caller can print the console table without touching the file again.
    """
    models: dict[str, dict] = {}

    for row in rows:
        m = row["model"]
        if m not in models:
            models[m] = {
                "n": 0, "em_sum": 0,
                "rougeL_sum": 0.0, "bert_sum": 0.0,
                "llm10_sum": 0.0, "llm5_sum": 0.0,
                "lat_sum": 0.0, "errors": 0, "subjects": set(),
            }
        d = models[m]
        d["n"]          += 1
        d["em_sum"]     += int(row.get("exact_match", 0) or 0)
        d["rougeL_sum"] += _safe_float(row.get("rouge_l"))
        d["bert_sum"]   += _safe_float(row.get("bert_score_f1"))
        d["llm10_sum"]  += _safe_float(row.get("llm_score_0_10"))
        d["llm5_sum"]   += _safe_float(row.get("llm_score_1_5"))
        d["lat_sum"]    += _safe_float(row.get("latency_s"))
        d["errors"]     += 1 if row.get("judge_error") else 0
        d["subjects"].add(row.get("subject", ""))

    ranked = sorted(
        models.items(),
        key=lambda kv: kv[1]["llm5_sum"] / max(kv[1]["n"], 1),
        reverse=True,
    )

    out_rows: list[dict] = []
    for rank, (model, d) in enumerate(ranked, 1):
        n = d["n"]
        out_rows.append({
            "rank":               rank,
            "model":              model,
            "n_questions":        n,
            "exact_match_rate":   f"{d['em_sum']/n:.4f}",
            "avg_rouge_l":        f"{d['rougeL_sum']/n:.4f}",
            "avg_bert_score_f1":  f"{d['bert_sum']/n:.4f}",
            "avg_llm_score_1_5":  f"{d['llm5_sum']/n:.4f}",
            "avg_llm_score_0_10": f"{d['llm10_sum']/n:.4f}",
            "avg_latency_s":      f"{d['lat_sum']/n:.3f}",
            "n_errors":           d["errors"],
            "subjects":           "|".join(sorted(d["subjects"])),
        })

    with open(LEADERBOARD_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    return out_rows


def write_subject_leaderboard(rows: list[dict]) -> list[dict]:
    """Compute per-(model, subject) aggregates from in-memory rows.

    Writes subject_leaderboard.csv and returns the ranked list of row dicts so
    the caller can print the console table without touching the file again.
    """
    groups: dict[tuple[str, str], dict] = {}

    for row in rows:
        key = (row["model"], row.get("subject", ""))
        if key not in groups:
            groups[key] = {
                "n": 0, "em_sum": 0,
                "rougeL_sum": 0.0, "bert_sum": 0.0,
                "llm5_sum": 0.0, "llm10_sum": 0.0,
            }
        d = groups[key]
        d["n"]          += 1
        d["em_sum"]     += int(row.get("exact_match", 0) or 0)
        d["rougeL_sum"] += _safe_float(row.get("rouge_l"))
        d["bert_sum"]   += _safe_float(row.get("bert_score_f1"))
        d["llm5_sum"]   += _safe_float(row.get("llm_score_1_5"))
        d["llm10_sum"]  += _safe_float(row.get("llm_score_0_10"))

    # Sort: model asc, then subject asc
    ranked = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    out_rows: list[dict] = []
    for rank, ((model, subject), d) in enumerate(ranked, 1):
        n = d["n"]
        out_rows.append({
            "rank":               rank,
            "model":              model,
            "subject":            subject,
            "n_questions":        n,
            "exact_match_rate":   f"{d['em_sum']/n:.4f}",
            "avg_rouge_l":        f"{d['rougeL_sum']/n:.4f}",
            "avg_bert_score_f1":  f"{d['bert_sum']/n:.4f}",
            "avg_llm_score_1_5":  f"{d['llm5_sum']/n:.4f}",
            "avg_llm_score_0_10": f"{d['llm10_sum']/n:.4f}",
        })

    with open(SUBJECT_LEADERBOARD_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBJECT_LEADERBOARD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    return out_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("EduBench-Local  |  Step 3: Evaluate")
    print("=" * 60)
    print(f"  Judge model   : {config.JUDGE_MODEL}")
    print(f"  Input         : {RAW_ANSWERS_PATH}")
    print(f"  Scored CSV    : {SCORED_RESULTS_PATH}")
    print(f"  Leaderboard   : {LEADERBOARD_PATH}")
    print(f"  Subject LB    : {SUBJECT_LEADERBOARD_PATH}")
    print()

    # -- Load raw answers -------------------------------------------------
    if not RAW_ANSWERS_PATH.exists():
        print(
            f"ERROR: {RAW_ANSWERS_PATH} not found.\n"
            "       Run step2_generate_answers.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(RAW_ANSWERS_PATH, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)
    print(f"Loaded {len(raw)} raw answers.")

    # -- Model filter ---------------------------------------------------------
    if TARGET_MODEL is not None:
        all_models = sorted({r["model"] for r in raw})
        raw = [r for r in raw if r["model"] == TARGET_MODEL]
        skipped = [m for m in all_models if m != TARGET_MODEL]
        print(f"  Filtering to model : {TARGET_MODEL}")
        if skipped:
            print(f"  Skipped models     : {', '.join(skipped)}")
        print(f"  Records after filter: {len(raw)}")
        if not raw:
            print(f"ERROR: no answers found for model '{TARGET_MODEL}'.",
                  file=sys.stderr)
            sys.exit(1)
        print()

    # -- Resumability -----------------------------------------------------
    done    = load_existing_scored()
    pending = [r for r in raw if (r["model"], r["id"]) not in done]
    print(f"Already scored: {len(done)} | Remaining: {len(pending)}\n")

    if not pending:
        print("All records already scored.")
    else:
        # -- Verify judge model is available ------------------------------
        try:
            ollama.show(config.JUDGE_MODEL)
        except Exception as e:
            err = str(e).lower()
            if "not found" in err or "pull" in err:
                print(
                    f"ERROR: Judge model '{config.JUDGE_MODEL}' not found locally.\n"
                    f"       Run:  ollama pull {config.JUDGE_MODEL}",
                    file=sys.stderr,
                )
            else:
                print(f"ERROR connecting to Ollama: {e}", file=sys.stderr)
                print("       Make sure Ollama is running: ollama serve", file=sys.stderr)
            sys.exit(1)

        # -- Open CSV (append if resuming, else write fresh header) -------
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_file, csv_writer = open_scored_csv(append=bool(done))

        # -- Phase 1: EM + ROUGE-L + LLM judge (per-record) --------------
        judge_errors = 0
        em_total     = 0
        rougeL_sum   = 0.0
        llm5_sum     = 0.0

        try:
            pbar = tqdm(
                pending,
                desc="  Scoring (EM+ROUGE+LLM)",
                unit="ans",
                dynamic_ncols=True,
                colour="green",
            )
            for rec in pbar:
                em    = exact_match(rec["reference_answer"], rec["student_answer"])
                rl    = rouge_l(rec["reference_answer"], rec["student_answer"])
                judge = _call_judge(rec["reference_answer"], rec["student_answer"])

                em_total   += em
                rougeL_sum += rl
                llm5_sum   += judge["score_1_5"]
                if judge["error"]:
                    judge_errors += 1

                row = {
                    "model":             rec["model"],
                    "id":                rec["id"],
                    "subject":           rec["subject"],
                    "question":          rec["question"],
                    "reference_answer":  rec["reference_answer"],
                    "student_answer":    rec["student_answer"],
                    "exact_match":       em,
                    "rouge_l":           rl,
                    "bert_score_f1":     "",   # filled in Phase 2
                    "llm_score_0_10":    judge["score_0_10"],
                    "llm_score_1_5":     judge["score_1_5"],
                    "llm_reasoning":     judge["reasoning"],
                    "latency_s":         rec.get("latency_s", ""),
                    "prompt_tokens":     rec.get("prompt_tokens", ""),
                    "completion_tokens": rec.get("completion_tokens", ""),
                    "total_tokens":      rec.get("total_tokens", ""),
                    "judge_error":       judge["error"] or "",
                }
                csv_writer.writerow(row)
                csv_file.flush()

                n_done = pbar.n + 1   # +1 because pbar.n is the count *before* this item
                pbar.set_postfix({
                    "EM":       f"{em_total/n_done:.0%}",
                    "ROUGE-L":  f"{rougeL_sum/n_done:.3f}",
                    "LLM(1-5)": f"{llm5_sum/n_done:.2f}",
                    "jerr":     judge_errors,
                })
        finally:
            csv_file.close()

        n_scored = len(pending)
        print(
            f"\n  Phase 1 done — {n_scored} answers scored | "
            f"EM: {em_total/n_scored:.1%} | "
            f"ROUGE-L: {rougeL_sum/n_scored:.3f} | "
            f"Avg LLM (1-5): {llm5_sum/n_scored:.2f} | "
            f"Judge errors: {judge_errors}"
        )

    # -- Phase 2: BERTScore (batch over all rows already in memory) -------
    print("\n  Phase 2: BERTScore (batched over all scored records) ...")

    # Load the full scored set from the file so previously-scored rows (from
    # earlier runs) are included alongside any rows scored this run.
    all_rows = _read_scored_rows()

    if not all_rows:
        print("  No scored rows found — skipping BERTScore and leaderboards.")
    else:
        # Only compute BERTScore for rows that don't have it yet
        needs_bert = [r for r in all_rows if not r.get("bert_score_f1")]
        if needs_bert:
            refs   = [r["reference_answer"] for r in needs_bert]
            cands  = [r["student_answer"]   for r in needs_bert]
            scores = compute_bert_scores(refs, cands)

            # Patch bert_score_f1 directly into the in-memory dicts
            for rec, s in zip(needs_bert, scores):
                rec["bert_score_f1"] = s

            # Rewrite the CSV once with all scores filled in
            with open(SCORED_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SCORED_FIELDNAMES)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"  BERTScore computed for {len(needs_bert):,} records.")
        else:
            print("  BERTScore already present for all records — skipping.")

        # -- Leaderboards: built entirely from the in-memory row list -----
        print("\nGenerating leaderboards ...")
        lb_rows  = write_leaderboard(all_rows)
        slb_rows = write_subject_leaderboard(all_rows)

        # -- Print overall leaderboard to console -------------------------
        print()
        hdr = (f"{'#':<4} {'Model':<22} {'EM%':>7} {'ROUGE-L':>8} "
               f"{'BERT-F1':>8} {'LLM(1-5)':>9} {'LLM(0-10)':>10} {'Lat':>7} {'N':>5}")
        print(hdr)
        print("-" * len(hdr))
        for row in lb_rows:
            print(
                f"  {row['rank']:<2} "
                f"{row['model']:<22} "
                f"{float(row['exact_match_rate'])*100:>6.1f}% "
                f"{float(row['avg_rouge_l']):>8.3f} "
                f"{float(row['avg_bert_score_f1']):>8.3f} "
                f"{float(row['avg_llm_score_1_5']):>9.2f} "
                f"{float(row['avg_llm_score_0_10']):>10.2f} "
                f"{float(row['avg_latency_s']):>6.2f}s "
                f"{row['n_questions']:>5}"
            )

        # -- Print subject leaderboard to console -------------------------
        print()
        print("Subject breakdown:")
        shdr = (f"  {'Model':<20} {'Subject':<35} {'EM%':>7} "
                f"{'ROUGE-L':>8} {'BERT-F1':>8} {'LLM(1-5)':>9} {'N':>5}")
        print(shdr)
        print("  " + "-" * (len(shdr) - 2))
        for row in slb_rows:
            print(
                f"  {row['model']:<20} "
                f"{row['subject']:<35} "
                f"{float(row['exact_match_rate'])*100:>6.1f}% "
                f"{float(row['avg_rouge_l']):>8.3f} "
                f"{float(row['avg_bert_score_f1']):>8.3f} "
                f"{float(row['avg_llm_score_1_5']):>9.2f} "
                f"{row['n_questions']:>5}"
            )

    print(f"\nScored results      : {SCORED_RESULTS_PATH}")
    print(f"Leaderboard         : {LEADERBOARD_PATH}")
    print(f"Subject leaderboard : {SUBJECT_LEADERBOARD_PATH}")
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
