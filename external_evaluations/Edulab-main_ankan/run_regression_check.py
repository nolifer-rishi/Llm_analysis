"""
run_regression_check.py - EduBench-Local CI / Regression Gate
===============================================================
Acts as a lightweight continuous-integration check for the full EduBench
pipeline.  Run this after any pipeline step to verify that:

  1. ARTIFACT CHECK  - All required output files and figures exist and are
                       non-empty (non-zero bytes).

  2. PERFORMANCE CHECK - Every model in leaderboard.csv meets minimum quality
                         thresholds:
                           - avg_llm_score_1_5  >= MIN_LLM_SCORE_1_5
                           - exact_match_rate   >= MIN_EXACT_MATCH_RATE
                           - avg_latency_s      <= MAX_AVG_LATENCY_S
                           - n_errors           <= MAX_JUDGE_ERRORS

  3. MODEL PRESENCE CHECK - The primary model (PRIMARY_MODEL) must appear in
                            leaderboard.csv with a sufficient number of
                            evaluated questions (>= MIN_QUESTIONS).

Exit codes:
  0  - All checks passed  (green light for CI)
  1  - One or more checks failed (red light; pipeline is degraded)

Usage:
  python run_regression_check.py

Thresholds (edit below or override via environment variables):
  EDUBENCH_MIN_LLM_SCORE    - float, default 3.0  (1-5 scale)
  EDUBENCH_MIN_EM_RATE      - float, default 0.00  (0-1 scale)
  EDUBENCH_MAX_LATENCY      - float, default 10.0  (seconds)
  EDUBENCH_MAX_ERRORS       - int,   default 10    (absolute count)
  EDUBENCH_MIN_QUESTIONS    - int,   default 10    (per model)
"""

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config

# ---------------------------------------------------------------------------
# Thresholds  (override via env-vars for CI pipelines)
# ---------------------------------------------------------------------------
PRIMARY_MODEL       : str   = "qwen2.5:3b"
MIN_LLM_SCORE_1_5   : float = float(os.environ.get("EDUBENCH_MIN_LLM_SCORE",  "3.0"))
MIN_EXACT_MATCH_RATE: float = float(os.environ.get("EDUBENCH_MIN_EM_RATE",     "0.00"))
MAX_AVG_LATENCY_S   : float = float(os.environ.get("EDUBENCH_MAX_LATENCY",    "10.0"))
MAX_JUDGE_ERRORS    : int   = int(  os.environ.get("EDUBENCH_MAX_ERRORS",     "10"))
MIN_QUESTIONS       : int   = int(  os.environ.get("EDUBENCH_MIN_QUESTIONS",  "10"))

# ---------------------------------------------------------------------------
# Required pipeline artifacts
# ---------------------------------------------------------------------------
REQUIRED_FILES: list[Path] = [
    config.RESULTS_DIR / "raw_answers.json",
    config.RESULTS_DIR / "scored_results.csv",
    config.RESULTS_DIR / "leaderboard.csv",
]

REQUIRED_FIGURES: list[str] = [
    "01_overall_score_bars.png",
    "02_latency_vs_accuracy.png",
    "03_subject_heatmap.png",
    "04_subject_bar_comparison.png",
    "05_exact_match_rate.png",
    "06_subject_metric_heatmap.png",   # always generated from subject_leaderboard
    "07_score_distribution.png",       # requires scored_results.csv
    "08_latency_distribution.png",     # requires scored_results.csv
    "09_tokens_vs_latency.png",        # requires scored_results.csv
]

# ---------------------------------------------------------------------------
# ANSI colours (disabled automatically on Windows when not supported)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR", "0") == "1"

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _green(t):  return _c("92;1", t)
def _red(t):    return _c("91;1", t)
def _yellow(t): return _c("93;1", t)
def _cyan(t):   return _c("96;1", t)
def _bold(t):   return _c("1",    t)
def _dim(t):    return _c("2",    t)

PASS_TAG = _green("  PASS  ")
FAIL_TAG = _red("  FAIL  ")
WARN_TAG = _yellow("  WARN  ")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name:    str
    passed:  bool
    message: str
    detail:  Optional[str] = None

@dataclass
class Section:
    title:   str
    checks:  List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_fail(self) -> int:
        return len(self.checks) - self.n_pass


# ---------------------------------------------------------------------------
# Section 1 – Artifact existence & non-empty check
# ---------------------------------------------------------------------------

def check_artifacts() -> Section:
    sec = Section("Artifact Existence & Size")

    # Core pipeline files
    for path in REQUIRED_FILES:
        exists = path.exists()
        size   = path.stat().st_size if exists else 0
        ok     = exists and size > 0
        label  = path.relative_to(config.ROOT_DIR).as_posix()
        sec.checks.append(CheckResult(
            name    = label,
            passed  = ok,
            message = f"{size:,} bytes" if ok else (
                "file not found" if not exists else "file is empty (0 bytes)"
            ),
            detail  = None if ok else f"Run the appropriate pipeline step to generate {label}",
        ))

    # Figures
    for fig_name in REQUIRED_FIGURES:
        path   = config.FIGURES_DIR / fig_name
        exists = path.exists()
        size   = path.stat().st_size if exists else 0
        ok     = exists and size > 0
        label  = f"figures/{fig_name}"
        sec.checks.append(CheckResult(
            name    = label,
            passed  = ok,
            message = f"{size:,} bytes" if ok else (
                "file not found" if not exists else "file is empty (0 bytes)"
            ),
            detail  = None if ok else "Run step4_visualize.py to regenerate figures",
        ))

    return sec


# ---------------------------------------------------------------------------
# Section 2 – JSON integrity (raw_answers.json)
# ---------------------------------------------------------------------------

def check_json_integrity() -> Section:
    sec  = Section("JSON Integrity")
    path = config.RESULTS_DIR / "raw_answers.json"

    if not path.exists():
        sec.checks.append(CheckResult(
            name    = "raw_answers.json parseable",
            passed  = False,
            message = "file not found – skipping JSON check",
        ))
        return sec

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else -1
        sec.checks.append(CheckResult(
            name    = "raw_answers.json parseable",
            passed  = True,
            message = f"{count:,} answer records loaded",
        ))

        # Check required keys in first record
        if count > 0:
            required_keys = {"model", "id", "subject", "question",
                             "reference_answer", "student_answer", "latency_s"}
            missing = required_keys - data[0].keys()
            sec.checks.append(CheckResult(
                name    = "raw_answers.json schema",
                passed  = not missing,
                message = "all required fields present" if not missing
                          else f"missing fields: {', '.join(sorted(missing))}",
            ))

    except (json.JSONDecodeError, OSError) as e:
        sec.checks.append(CheckResult(
            name    = "raw_answers.json parseable",
            passed  = False,
            message = f"parse error: {e}",
        ))

    return sec


# ---------------------------------------------------------------------------
# Section 3 – Leaderboard CSV integrity
# ---------------------------------------------------------------------------

def check_csv_integrity() -> Section:
    sec  = Section("CSV Integrity")
    path = config.RESULTS_DIR / "leaderboard.csv"

    if not path.exists():
        sec.checks.append(CheckResult(
            name="leaderboard.csv readable", passed=False,
            message="file not found",
        ))
        return sec

    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        sec.checks.append(CheckResult(
            name    = "leaderboard.csv readable",
            passed  = True,
            message = f"{len(rows)} model(s) in leaderboard",
        ))

        required_cols = {
            "model", "n_questions", "exact_match_rate",
            "avg_llm_score_1_5", "avg_latency_s", "n_errors",
        }
        if rows:
            missing = required_cols - rows[0].keys()
            sec.checks.append(CheckResult(
                name    = "leaderboard.csv schema",
                passed  = not missing,
                message = "all required columns present" if not missing
                          else f"missing columns: {', '.join(sorted(missing))}",
            ))

    except (csv.Error, OSError) as e:
        sec.checks.append(CheckResult(
            name="leaderboard.csv readable", passed=False,
            message=f"CSV error: {e}",
        ))

    return sec


# ---------------------------------------------------------------------------
# Section 4 – Performance thresholds per model
# ---------------------------------------------------------------------------

def check_performance() -> Section:
    sec  = Section("Performance Thresholds")
    path = config.RESULTS_DIR / "leaderboard.csv"

    if not path.exists():
        sec.checks.append(CheckResult(
            name="leaderboard available", passed=False,
            message="leaderboard.csv not found – cannot check performance",
        ))
        return sec

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sec.checks.append(CheckResult(
            name="leaderboard non-empty", passed=False,
            message="leaderboard.csv has no model rows",
        ))
        return sec

    for row in rows:
        model = row["model"]
        try:
            n_q    = int(row["n_questions"])
            em     = float(row["exact_match_rate"])
            llm5   = float(row["avg_llm_score_1_5"])
            lat    = float(row["avg_latency_s"])
            n_err  = int(row["n_errors"])
        except (ValueError, KeyError) as e:
            sec.checks.append(CheckResult(
                name=f"[{model}] data parseable", passed=False,
                message=f"could not parse row: {e}",
            ))
            continue

        # -- Minimum question count
        sec.checks.append(CheckResult(
            name    = f"[{model}] question count >= {MIN_QUESTIONS}",
            passed  = n_q >= MIN_QUESTIONS,
            message = f"{n_q} questions evaluated",
            detail  = f"Only {n_q} questions; run step2 + step3 with more data."
                      if n_q < MIN_QUESTIONS else None,
        ))

        # -- LLM accuracy (1-5 scale)
        sec.checks.append(CheckResult(
            name    = f"[{model}] avg LLM score (1-5) >= {MIN_LLM_SCORE_1_5:.2f}",
            passed  = llm5 >= MIN_LLM_SCORE_1_5,
            message = f"actual: {llm5:.4f}",
            detail  = f"Score {llm5:.4f} is below threshold {MIN_LLM_SCORE_1_5:.2f}. "
                      "Model accuracy may have degraded."
                      if llm5 < MIN_LLM_SCORE_1_5 else None,
        ))

        # -- Exact match rate
        sec.checks.append(CheckResult(
            name    = f"[{model}] exact match rate >= {MIN_EXACT_MATCH_RATE:.2%}",
            passed  = em >= MIN_EXACT_MATCH_RATE,
            message = f"actual: {em:.4f}  ({em:.1%})",
            detail  = f"EM rate {em:.1%} is below threshold {MIN_EXACT_MATCH_RATE:.1%}."
                      if em < MIN_EXACT_MATCH_RATE else None,
        ))

        # -- Latency
        sec.checks.append(CheckResult(
            name    = f"[{model}] avg latency <= {MAX_AVG_LATENCY_S:.1f}s",
            passed  = lat <= MAX_AVG_LATENCY_S,
            message = f"actual: {lat:.3f}s",
            detail  = f"Latency {lat:.3f}s exceeds {MAX_AVG_LATENCY_S:.1f}s budget. "
                      "Check hardware load or model size."
                      if lat > MAX_AVG_LATENCY_S else None,
        ))

        # -- Judge errors
        sec.checks.append(CheckResult(
            name    = f"[{model}] judge errors <= {MAX_JUDGE_ERRORS}",
            passed  = n_err <= MAX_JUDGE_ERRORS,
            message = f"actual: {n_err}",
            detail  = f"{n_err} judge errors exceed threshold {MAX_JUDGE_ERRORS}. "
                      "Check Ollama connectivity and judge model availability."
                      if n_err > MAX_JUDGE_ERRORS else None,
        ))

    return sec


# ---------------------------------------------------------------------------
# Section 5 – Primary model presence
# ---------------------------------------------------------------------------

def check_primary_model() -> Section:
    sec  = Section(f"Primary Model Presence  [{PRIMARY_MODEL}]")
    path = config.RESULTS_DIR / "leaderboard.csv"

    if not path.exists():
        sec.checks.append(CheckResult(
            name="leaderboard available", passed=False,
            message="leaderboard.csv not found",
        ))
        return sec

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    models_found = [r["model"] for r in rows]
    found = PRIMARY_MODEL in models_found

    sec.checks.append(CheckResult(
        name    = f"'{PRIMARY_MODEL}' present in leaderboard",
        passed  = found,
        message = f"found — ranked #{models_found.index(PRIMARY_MODEL) + 1} of {len(models_found)}"
                  if found else f"not found. Models present: {models_found or ['(none)']}",
        detail  = f"Run step2 + step3 for model '{PRIMARY_MODEL}'."
                  if not found else None,
    ))

    if found:
        row = next(r for r in rows if r["model"] == PRIMARY_MODEL)
        try:
            n_q  = int(row["n_questions"])
            llm5 = float(row["avg_llm_score_1_5"])
            lat  = float(row["avg_latency_s"])
            sec.checks.append(CheckResult(
                name    = f"'{PRIMARY_MODEL}' meets minimum question count",
                passed  = n_q >= MIN_QUESTIONS,
                message = f"{n_q} questions  |  LLM(1-5): {llm5:.4f}  |  Latency: {lat:.3f}s",
                detail  = f"Only {n_q}/{MIN_QUESTIONS} required questions evaluated."
                          if n_q < MIN_QUESTIONS else None,
            ))
        except (ValueError, KeyError):
            pass

    return sec


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

SECTION_WIDTH = 72

def _rule(char: str = "-") -> str:
    return char * SECTION_WIDTH

def render_section(sec: Section) -> None:
    status = _green("PASSED") if sec.passed else _red("FAILED")
    print()
    print(_bold(_rule("=")))
    print(_bold(f"  {sec.title}") + "  " + _dim(f"({sec.n_pass}/{len(sec.checks)} checks passed)") + "  " + status)
    print(_rule("-"))

    for chk in sec.checks:
        tag = PASS_TAG if chk.passed else FAIL_TAG
        print(f"[{tag}]  {chk.name}")
        print(f"         {_dim(chk.message)}")
        if chk.detail and not chk.passed:
            print(f"         {_yellow('Hint:')} {chk.detail}")


def render_summary(sections: List[Section], elapsed_ms: float) -> int:
    total_checks = sum(len(s.checks) for s in sections)
    total_pass   = sum(s.n_pass for s in sections)
    total_fail   = total_checks - total_pass
    all_pass     = total_fail == 0

    print()
    print(_bold(_rule("=")))
    print(_bold("  REGRESSION GATE SUMMARY"))
    print(_rule("="))
    print()

    # Per-section summary table
    col_w = max(len(s.title) for s in sections) + 2
    for sec in sections:
        status_str = _green("PASS") if sec.passed else _red("FAIL")
        bar_filled = round(sec.n_pass / max(len(sec.checks), 1) * 20)
        bar = _green("#" * bar_filled) + _dim("-" * (20 - bar_filled))
        print(f"  {sec.title:<{col_w}}  {bar}  {status_str}  "
              f"{_dim(f'{sec.n_pass}/{len(sec.checks)}')}")

    print()
    print(_rule("-"))
    print(f"  Total checks  : {total_checks}")
    print(f"  Passed        : {_green(str(total_pass))}")
    print(f"  Failed        : {_red(str(total_fail)) if total_fail else _dim('0')}")
    print(f"  Elapsed       : {elapsed_ms:.0f} ms")
    print(f"  Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_rule("-"))

    if all_pass:
        print()
        print(_green("  [OK]  ALL REGRESSION CHECKS PASSED -- pipeline is healthy."))
    else:
        print()
        print(_red(f"  [!!]  REGRESSION GATE FAILED -- {total_fail} check(s) did not pass."))
        print(_yellow("        Review the FAIL items above and re-run the relevant pipeline step."))

    print(_rule("="))
    print()

    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import time
    t0 = time.perf_counter()

    print()
    print(_bold("=" * SECTION_WIDTH))
    print(_bold("  EduBench-Local  |  CI Regression Gate"))
    print(_bold("=" * SECTION_WIDTH))
    print(f"  Root          : {config.ROOT_DIR}")
    print(f"  Results dir   : {config.RESULTS_DIR}")
    print(f"  Figures dir   : {config.FIGURES_DIR}")
    print()
    print(_bold("  Thresholds:"))
    print(f"    Primary model         : {PRIMARY_MODEL}")
    print(f"    Min LLM score (1-5)   : {MIN_LLM_SCORE_1_5}")
    print(f"    Min exact match rate  : {MIN_EXACT_MATCH_RATE:.2%}")
    print(f"    Max avg latency       : {MAX_AVG_LATENCY_S}s")
    print(f"    Max judge errors      : {MAX_JUDGE_ERRORS}")
    print(f"    Min questions         : {MIN_QUESTIONS}")

    sections = [
        check_artifacts(),
        check_json_integrity(),
        check_csv_integrity(),
        check_primary_model(),
        check_performance(),
    ]

    for sec in sections:
        render_section(sec)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return render_summary(sections, elapsed_ms)


if __name__ == "__main__":
    sys.exit(main())
