"""
step1_prepare_dataset.py - EduBench-Local Pipeline Step 1
==========================================================
Loads five educational datasets via the Hugging Face Datasets Server
REST API (pure HTTP + JSON - no pyarrow/DLL needed):

  1. allenai/sciq            (SciQ)          split=test
  2. allenai/openbookqa      (OpenBookQA)    split=test
  3. allenai/ai2_arc Chall.  (ARC-Challenge) split=test
  4. ehovy/race / middle     (RACE-middle)   split=test
  5. rajpurkar/squad         (SQuAD v1.1)   split=validation

Standard schema per record:
  {
    "id":               str,
    "subject":          str,
    "question":         str,
    "context":          str,   # supporting passage or article excerpt
    "reference_answer": str
  }

Deterministic random sample: SAMPLE_SIZE_PER_SUBJECT=150, RANDOM_SEED=42
Output: data/dataset_sample.json  (750 records total)

Usage:
  python step1_prepare_dataset.py
"""

import json
import random
import sys
import time
import urllib.request
import urllib.error

import config

# ---------------------------------------------------------------------------
# HuggingFace Datasets Server API helpers
# ---------------------------------------------------------------------------

HF_API_BASE = "https://datasets-server.huggingface.co"


def hf_get_rows(dataset: str, config_name: str, split: str,
                offset: int = 0, length: int = 100) -> dict:
    """Fetch a page of rows from the HF Datasets Server."""
    url = (
        f"{HF_API_BASE}/rows"
        f"?dataset={dataset}"
        f"&config={config_name}"
        f"&split={split}"
        f"&offset={offset}"
        f"&length={length}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "EduBench-Local/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_all_rows(dataset: str, config_name: str, split: str,
                   page_size: int = 100, max_rows: int = 3000,
                   verbose: bool = True) -> list[dict]:
    """
    Page through the HF Datasets Server and collect raw rows.
    Stops after max_rows or when the server signals end-of-data.
    HTTP 4xx (except 429) is treated as end-of-data (not a fatal error).
    HTTP 429 (rate-limit) is retried with exponential back-off.
    """
    MAX_RETRIES   = 5
    BACKOFF_START = 5.0   # seconds; doubles each retry

    rows = []
    offset = 0
    while len(rows) < max_rows:
        to_fetch = min(page_size, max_rows - len(rows))
        retries  = 0
        while True:
            try:
                result = hf_get_rows(dataset, config_name, split, offset, to_fetch)
                break  # success
            except urllib.error.HTTPError as e:
                if e.code == 429 and retries < MAX_RETRIES:
                    wait = BACKOFF_START * (2 ** retries)
                    if verbose:
                        print(f"\n    429 rate-limited; retrying in {wait:.0f}s "
                              f"(attempt {retries + 1}/{MAX_RETRIES}) ...",
                              flush=True)
                    time.sleep(wait)
                    retries += 1
                elif e.code in (404, 416):
                    # Offset beyond server-exposed range; treat as end-of-data
                    if verbose:
                        print()
                    return rows
                else:
                    raise
        batch = result.get("rows", [])
        if not batch:
            break
        rows.extend(r["row"] for r in batch)
        if verbose:
            print(f"    fetched {len(rows):>5} rows ...", end="\r", flush=True)
        offset += len(batch)
        if len(batch) < to_fetch:
            break   # reached actual end
        time.sleep(0.3)  # 300 ms between pages to avoid rate-limiting
    if verbose:
        print()
    return rows


# ---------------------------------------------------------------------------
# Deterministic sampling helper
# ---------------------------------------------------------------------------

def _sample(items: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(items, min(n, len(items)))


# ---------------------------------------------------------------------------
# Formatters  (one per dataset)
# ---------------------------------------------------------------------------

def format_sciq(rows: list[dict]) -> list[dict]:
    """
    allenai/sciq fields:
      question, correct_answer, support (passage, may be empty),
      distractor1/2/3
    """
    out = []
    for i, row in enumerate(rows):
        out.append({
            "id":               f"sciq_{i}",
            "subject":          "science",
            "question":         row.get("question", "").strip(),
            "context":          row.get("support", "").strip(),
            "reference_answer": row.get("correct_answer", "").strip(),
        })
    return out


def format_openbookqa(rows: list[dict]) -> list[dict]:
    """
    openbookqa fields:
      id, question_stem, choices {text: [...], label: [...]},
      answerKey (A/B/C/D)
    """
    out = []
    for i, row in enumerate(rows):
        choices    = row.get("choices", {})
        texts      = choices.get("text", [])
        labels     = choices.get("label", [])
        answer_key = row.get("answerKey", "A").strip()
        ref = answer_key
        for label, text in zip(labels, texts):
            if label.strip() == answer_key:
                ref = text.strip()
                break
        out.append({
            "id":               f"openbookqa_{i}",
            "subject":          "general_science",
            "question":         row.get("question_stem", "").strip(),
            "context":          "",   # OpenBookQA has no passage context
            "reference_answer": ref,
        })
    return out


def _format_arc(rows: list[dict], tag: str, subject: str) -> list[dict]:
    """
    Shared formatter for ARC-Easy and ARC-Challenge.
    ai2_arc fields:
      id, question, choices {text: [...], label: [...]}, answerKey
    """
    out = []
    for i, row in enumerate(rows):
        choices    = row.get("choices", {})
        texts      = choices.get("text", [])
        labels     = choices.get("label", [])
        answer_key = row.get("answerKey", "A").strip()
        ref = answer_key
        for label, text in zip(labels, texts):
            if label.strip() == answer_key:
                ref = text.strip()
                break
        out.append({
            "id":               f"{tag}_{i}",
            "subject":          subject,
            "question":         row.get("question", "").strip(),
            "context":          "",   # ARC has no passage context
            "reference_answer": ref,
        })
    return out


def format_arc_easy(rows: list[dict]) -> list[dict]:
    return _format_arc(rows, tag="arc_easy", subject="science_easy")


def format_arc_challenge(rows: list[dict]) -> list[dict]:
    return _format_arc(rows, tag="arc_challenge", subject="science_challenge")


def format_race(rows: list[dict]) -> list[dict]:
    """
    race/middle fields:
      article, question, answer (A/B/C/D), options (list of 4)
    """
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    out = []
    for i, row in enumerate(rows):
        letter  = row.get("answer", "A").strip()
        idx     = letter_to_idx.get(letter, 0)
        options = row.get("options", [])
        ref     = options[idx].strip() if idx < len(options) else letter
        out.append({
            "id":               f"race_middle_{i}",
            "subject":          "reading_comprehension",
            "question":         row.get("question", "").strip(),
            "context":          row.get("article", "").strip(),
            "reference_answer": ref,
        })
    return out


def format_squad(rows: list[dict]) -> list[dict]:
    """
    rajpurkar/squad fields:
      id, title, context, question,
      answers: {text: [...], answer_start: [...]}
    """
    out = []
    for i, row in enumerate(rows):
        answers = row.get("answers", {})
        texts   = answers.get("text", [])
        ref     = texts[0].strip() if texts else ""
        out.append({
            "id":               f"squad_{i}",
            "subject":          "reading_comprehension_squad",
            "question":         row.get("question", "").strip(),
            "context":          row.get("context", "").strip(),
            "reference_answer": ref,
        })
    return out


# ---------------------------------------------------------------------------
# Dataset manifest  (order, label, fetch params, formatter)
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "label":     "SciQ (allenai/sciq)",
        "dataset":   "allenai/sciq",
        "config":    "default",
        "split":     "test",
        "formatter": format_sciq,
    },
    {
        "label":     "OpenBookQA",
        "dataset":   "allenai/openbookqa",
        "config":    "main",
        "split":     "test",
        "formatter": format_openbookqa,
    },
    {
        "label":     "ARC-Challenge",
        "dataset":   "allenai/ai2_arc",
        "config":    "ARC-Challenge",
        "split":     "test",
        "formatter": format_arc_challenge,
    },
    {
        "label":     "RACE-middle",
        "dataset":   "ehovy/race",
        "config":    "middle",
        "split":     "test",
        "formatter": format_race,
    },
    {
        "label":     "SQuAD v1.1",
        "dataset":   "rajpurkar/squad",
        "config":    "plain_text",
        "split":     "validation",
        "formatter": format_squad,
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n    = config.SAMPLE_SIZE_PER_SUBJECT   # 150
    seed = config.RANDOM_SEED               # 42
    out  = config.DATASET_SAMPLE_PATH

    print("=" * 60)
    print("EduBench-Local  |  Step 1: Prepare Dataset")
    print("=" * 60)
    print(f"  Datasets                : {len(DATASETS)}")
    print(f"  SAMPLE_SIZE_PER_SUBJECT : {n}")
    print(f"  RANDOM_SEED             : {seed}")
    print(f"  Output path             : {out}")
    print(f"  Fetching via            : HF Datasets Server REST API")
    print()

    combined: list[dict] = []
    total_datasets = len(DATASETS)

    for idx, ds in enumerate(DATASETS, start=1):
        label   = ds["label"]
        dataset = ds["dataset"]
        cfg     = ds["config"]
        split   = ds["split"]
        fmt_fn  = ds["formatter"]

        print(f"[{idx}/{total_datasets}] Fetching {label} ...")
        try:
            rows = fetch_all_rows(
                dataset=dataset,
                config_name=cfg,
                split=split,
                max_rows=max(n * 8, 1200),
            )
        except Exception as exc:
            print(f"  FATAL: {exc}", file=sys.stderr)
            sys.exit(1)

        if not rows:
            print(
                f"  ERROR: got 0 rows from {label} - "
                "check internet connection.",
                file=sys.stderr,
            )
            sys.exit(1)

        all_records = fmt_fn(rows)
        sample      = _sample(all_records, n, seed)
        print(f"  Fetched {len(all_records):,} records -> sampled {len(sample):,}")
        combined.extend(sample)

    # -- Save -----------------------------------------------------------------
    expected = n * total_datasets
    print(f"\nCombined dataset size : {len(combined):,} records "
          f"(expected ~{expected:,})")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # Sanity check
    with open(out, encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == len(combined), (
        f"Write/read mismatch: wrote {len(combined)}, read {len(loaded)}"
    )

    print(f"Saved  -> {out}")

    # -- Sample record preview ------------------------------------------------
    print("\nSample record (first):")
    s = loaded[0]
    print(f"  id               : {s['id']}")
    print(f"  subject          : {s['subject']}")
    q = s["question"]
    print(f"  question         : {q[:80]}{'...' if len(q) > 80 else ''}")
    ctx = s["context"]
    if ctx:
        print(f"  context          : {ctx[:60]}{'...' if len(ctx) > 60 else ''}")
    else:
        print(f"  context          : (none)")
    ref = s["reference_answer"]
    print(f"  reference_answer : {ref[:80]}{'...' if len(ref) > 80 else ''}")

    # -- Dataset breakdown summary -------------------------------------------
    print("\nDataset breakdown:")
    subjects: dict[str, int] = {}
    for rec in loaded:
        subjects[rec["subject"]] = subjects.get(rec["subject"], 0) + 1
    for subj, count in subjects.items():
        print(f"  {subj:<35} : {count:>4} records")

    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()

