"""
api.py - EduBench-Local FastAPI Backend Service
================================================
A production-ready async REST API that exposes EduBench-Local data and
live Ollama inference over HTTP.

Endpoints
---------
  GET  /             Redirect to interactive docs
  GET  /health       Liveness probe: API + Ollama status, model inventory
  GET  /leaderboard  Full + per-subject leaderboard (JSON)
  POST /generate     Async Ollama generation with latency & token metrics

CORS
----
All origins are allowed for local development.  Tighten `allow_origins`
before deploying to production (see CORS_ORIGINS below).

Usage
-----
  # development (auto-reload)
  edubench-env\\Scripts\\uvicorn.exe api:app --reload --port 8000

  # production
  edubench-env\\Scripts\\uvicorn.exe api:app --host 0.0.0.0 --port 8000 --workers 2

Interactive docs
----------------
  http://localhost:8000/docs   (Swagger UI)
  http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import config

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EduBench-Local API",
    description=(
        "REST backend for the EduBench-Local LLM benchmarking pipeline.\n\n"
        "Exposes leaderboard metrics and live Ollama inference in a single service."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — open for local development; restrict in production
# ---------------------------------------------------------------------------
CORS_ORIGINS: list[str] = [
    "http://localhost",
    "http://localhost:8501",   # Streamlit default
    "http://localhost:8502",   # Streamlit alt port
    "http://localhost:3000",   # React / Next.js dev
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8502",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEADERBOARD_PATH         = config.RESULTS_DIR / "leaderboard.csv"
SUBJECT_LEADERBOARD_PATH = config.RESULTS_DIR / "subject_leaderboard.csv"
OLLAMA_BASE_URL: str     = config.OLLAMA_BASE_URL   # e.g. http://localhost:11434


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HealthOllama(BaseModel):
    reachable:    bool
    version:      str | None = None
    local_models: list[str]  = Field(default_factory=list)
    error:        str | None = None


class HealthResponse(BaseModel):
    status:     str          # "ok" | "degraded"
    api:        str = "ok"
    ollama:     HealthOllama
    results_dir_exists: bool
    figures_dir_exists: bool


class LeaderboardRow(BaseModel):
    model:               str
    n_questions:         int
    exact_match_rate:    float
    avg_llm_score_1_5:   float
    avg_llm_score_0_10:  float
    avg_latency_s:       float
    n_errors:            int
    subjects:            list[str]


class SubjectRow(BaseModel):
    model:               str
    subject:             str
    n_questions:         int
    exact_match_rate:    float
    avg_rouge_l:         float | None = None
    avg_bert_score_f1:   float | None = None
    avg_llm_score_1_5:   float
    avg_llm_score_0_10:  float | None = None


class LeaderboardResponse(BaseModel):
    overall:  list[LeaderboardRow]
    by_subject: list[SubjectRow]


class GenerateRequest(BaseModel):
    model:       str   = Field(
        default="qwen2.5:3b",
        examples=["qwen2.5:3b", "mistral:7b"],
        description="Ollama model tag (must be pulled locally)",
    )
    prompt:      str   = Field(
        ...,
        min_length=1,
        examples=["Explain Newton's second law of motion."],
        description="The question or instruction to send to the model",
    )
    subject:     str | None = Field(
        default=None,
        examples=["Science"],
        description="Optional subject context injected into the system prompt",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens:  int   = Field(default=200, ge=1, le=4096)


class GenerateResponse(BaseModel):
    model:             str
    prompt:            str
    answer:            str
    latency_s:         float
    prompt_tokens:     int | None = None
    completion_tokens: int | None = None
    total_tokens:      int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_to_dicts(path: Path) -> list[dict[str, Any]]:
    """Read a CSV and return a list of row dicts.  Returns [] if file missing."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_leaderboard(rows: list[dict]) -> list[LeaderboardRow]:
    out: list[LeaderboardRow] = []
    for row in rows:
        try:
            out.append(LeaderboardRow(
                model              = row["model"],
                n_questions        = int(row["n_questions"]),
                exact_match_rate   = float(row["exact_match_rate"]),
                avg_llm_score_1_5  = float(row["avg_llm_score_1_5"]),
                avg_llm_score_0_10 = float(row["avg_llm_score_0_10"]),
                avg_latency_s      = float(row["avg_latency_s"]),
                n_errors           = int(row["n_errors"]),
                subjects           = row["subjects"].split("|") if row.get("subjects") else [],
            ))
        except (KeyError, ValueError):
            continue
    return out


def _parse_subject_leaderboard(rows: list[dict]) -> list[SubjectRow]:
    out: list[SubjectRow] = []
    for row in rows:
        try:
            out.append(SubjectRow(
                model             = row["model"],
                subject           = row["subject"],
                n_questions       = int(row["n_questions"]),
                exact_match_rate  = float(row["exact_match_rate"]),
                avg_rouge_l       = float(row["avg_rouge_l"])       if row.get("avg_rouge_l")       else None,
                avg_bert_score_f1 = float(row["avg_bert_score_f1"]) if row.get("avg_bert_score_f1") else None,
                avg_llm_score_1_5 = float(row["avg_llm_score_1_5"]),
                avg_llm_score_0_10= float(row["avg_llm_score_0_10"]) if row.get("avg_llm_score_0_10") else None,
            ))
        except (KeyError, ValueError):
            continue
    return out


async def _probe_ollama() -> HealthOllama:
    """Async HTTP probe to the Ollama daemon.  Never raises."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            # /api/version — lightweight version check
            ver_resp  = await client.get(f"{OLLAMA_BASE_URL}/api/version")
            version   = ver_resp.json().get("version") if ver_resp.is_success else None

            # /api/tags — list local models
            tags_resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models: list[str] = []
            if tags_resp.is_success:
                for m in tags_resp.json().get("models", []):
                    name = m.get("name") or m.get("model", "")
                    if name:
                        models.append(name)

        return HealthOllama(reachable=True, version=version, local_models=models)

    except Exception as exc:
        return HealthOllama(reachable=False, error=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect bare root to the interactive Swagger docs."""
    return RedirectResponse(url="/docs")


# ── GET /health ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"],
)
async def health() -> HealthResponse:
    """
    Liveness / readiness probe.

    Returns:
    - **status**: `"ok"` if Ollama is reachable, `"degraded"` otherwise.
    - **ollama**: version string, reachability flag, and list of local model tags.
    - **results_dir_exists** / **figures_dir_exists**: pipeline artifact sanity check.
    """
    ollama_info  = await _probe_ollama()
    overall      = "ok" if ollama_info.reachable else "degraded"

    return HealthResponse(
        status              = overall,
        ollama              = ollama_info,
        results_dir_exists  = config.RESULTS_DIR.exists(),
        figures_dir_exists  = config.FIGURES_DIR.exists(),
    )


# ── GET /leaderboard ─────────────────────────────────────────────────────────

@app.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    summary="Get leaderboard data",
    tags=["Leaderboard"],
)
async def leaderboard() -> LeaderboardResponse:
    """
    Return aggregated benchmark results.

    - **overall**: one entry per model (from `results/leaderboard.csv`)
    - **by_subject**: one entry per model × subject (from `results/subject_leaderboard.csv`)

    Raises **404** if neither CSV exists (step 3 has not been run yet).
    """
    overall_rows = _csv_to_dicts(LEADERBOARD_PATH)
    subject_rows = _csv_to_dicts(SUBJECT_LEADERBOARD_PATH)

    if not overall_rows and not subject_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Leaderboard CSVs not found. "
                "Run step3_evaluate.py (and optionally step4_visualize.py) first."
            ),
        )

    return LeaderboardResponse(
        overall    = _parse_leaderboard(overall_rows),
        by_subject = _parse_subject_leaderboard(subject_rows),
    )


# ── POST /generate ────────────────────────────────────────────────────────────

@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate a model answer via Ollama",
    tags=["Inference"],
)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Send a prompt to a local Ollama model and return the generated answer.

    **Request body**:
    - `model` — Ollama model tag (default: `qwen2.5:3b`)
    - `prompt` — The question or instruction
    - `subject` — Optional subject label injected as context (e.g. `"Science"`)
    - `temperature` — Sampling temperature (0.0 – 2.0, default 0.2)
    - `max_tokens` — Maximum tokens to generate (default 200)

    **Response**:
    - `answer` — The model's response text
    - `latency_s` — Wall-clock seconds for the Ollama round-trip
    - `prompt_tokens`, `completion_tokens`, `total_tokens` — token counts (when available)

    Raises **503** if Ollama is unreachable, **422** for bad input.
    """
    # Build the full prompt with optional subject context
    if req.subject:
        full_prompt = (
            f"You are a knowledgeable educational assistant.\n"
            f"Subject: {req.subject}\n\n"
            f"Question: {req.prompt}\n\n"
            f"Answer clearly and accurately:"
        )
    else:
        full_prompt = (
            f"You are a knowledgeable educational assistant.\n\n"
            f"Question: {req.prompt}\n\n"
            f"Answer clearly and accurately:"
        )

    # Async HTTP call to Ollama's /api/chat endpoint
    payload = {
        "model": req.model,
        "messages": [{"role": "user", "content": full_prompt}],
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
        "stream": False,
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                "Ensure Ollama is running: `ollama serve`"
            ),
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Ollama did not respond within the timeout window (120 s).",
        )

    latency_s = time.perf_counter() - t0

    if not resp.is_success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama returned HTTP {resp.status_code}: {resp.text[:400]}",
        )

    data   = resp.json()
    answer = data.get("message", {}).get("content", "").strip()

    # Token metrics — Ollama returns these at the top level
    prompt_tokens     = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    total_tokens      = (
        (prompt_tokens or 0) + (completion_tokens or 0)
    ) or None

    return GenerateResponse(
        model             = req.model,
        prompt            = req.prompt,
        answer            = answer,
        latency_s         = round(latency_s, 4),
        prompt_tokens     = prompt_tokens,
        completion_tokens = completion_tokens,
        total_tokens      = total_tokens,
    )


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
