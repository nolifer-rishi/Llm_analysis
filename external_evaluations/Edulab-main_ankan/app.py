# -*- coding: utf-8 -*-
"""
app.py - EduBench-Local Interactive Dashboard
==============================================
A Streamlit web dashboard for the EduBench-Local benchmarking pipeline.

Features:
  - Multi-model evaluation support: interactive dropdown/toggle between
    "Qwen 2.5 3B" and "Google Gemini", or a side-by-side comparison view.
  - Live data loading: dynamically switches between results/scored_results.csv (Qwen)
    and results/gemini_scored_results.csv (Gemini).
  - Visualizations gallery: showcases all 11 comparative figures and scorecards.
  - Live model playground: test queries in real-time with latency tracking.
  - Auto-launcher: launches Streamlit automatically when run via `python app.py`.

Usage:
  python app.py
  # or: streamlit run app.py
"""

from __future__ import annotations

import io
import os
import sys

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError in Click/Streamlit
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Auto-launcher: if executed via `python app.py`, launch Streamlit automatically
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _under_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _under_streamlit = (get_script_run_ctx() is not None)
    except Exception:
        pass

    if not _under_streamlit:
        from streamlit.web import cli as stcli
        target_script = os.path.abspath(__file__)
        sys.argv = ["streamlit", "run", target_script] + sys.argv[1:]
        sys.exit(stcli.main())


import time
from pathlib import Path
from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

import config

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduBench-Local Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark premium design
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Core background ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b2e 100%);
    }
    [data-testid="stSidebar"] {
        background: #0f1623;
        border-right: 1px solid #1e2a45;
    }

    /* ── Typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #e2e8f0;
    }
    h1 { color: #60a5fa; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #93c5fd; font-weight: 700; }
    h3 { color: #bfdbfe; font-weight: 600; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e2d4a, #162040);
        border: 1px solid #2d4a7a;
        border-radius: 14px;
        padding: 18px 20px 14px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    [data-testid="metric-container"] label {
        color: #7aafff !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1e2a45;
    }

    /* ── Sidebar nav ── */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        font-weight: 500;
        padding: 6px 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(37,99,235,0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        box-shadow: 0 6px 20px rgba(59,130,246,0.5);
        transform: translateY(-1px);
    }

    /* ── Select / text inputs ── */
    .stSelectbox > div > div,
    .stTextArea > div > div,
    .stTextInput > div > div {
        background: #1a2540 !important;
        border: 1px solid #2d4a7a !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ── Code / response box ── */
    .response-box {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d1fae5;
        white-space: pre-wrap;
        max-height: 420px;
        overflow-y: auto;
    }
    .latency-badge {
        display: inline-block;
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 10px;
        border: 1px solid #059669;
    }
    .error-box {
        background: #1f0a0a;
        border: 1px solid #7f1d1d;
        border-radius: 12px;
        padding: 16px 20px;
        color: #fca5a5;
        font-size: 0.9rem;
    }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d4a7a, transparent);
        margin: 28px 0;
    }

    /* ── Page hero ── */
    .page-hero {
        background: linear-gradient(135deg, #0f2447 0%, #162040 100%);
        border: 1px solid #1e3a6e;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .page-hero h1 { margin: 0 0 6px 0; font-size: 1.85rem; }
    .page-hero p  { color: #7aafff; margin: 0; font-size: 0.95rem; }

    /* ── Model badge pill ── */
    .model-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .pill-qwen { background: #1e3a6e; color: #60a5fa; border: 1px solid #3b82f6; }
    .pill-gemini { background: #45220c; color: #fb923c; border: 1px solid #f97316; }

    /* ── Figure card ── */
    .fig-card {
        background: #111827;
        border: 1px solid #1e2a45;
        border-radius: 14px;
        padding: 16px;
        transition: box-shadow 0.2s;
    }
    .fig-card:hover {
        box-shadow: 0 0 0 2px #3b82f6, 0 8px 32px rgba(59,130,246,0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers & Data Loaders
# ---------------------------------------------------------------------------
RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR

DATASET_NAME_MAP = {
    "science":                     "SciQ",
    "general_science":             "OpenBookQA",
    "science_challenge":           "ARC-Challenge",
    "reading_comprehension":       "RACE",
    "reading_comprehension_squad": "SQuAD v1.1",
}
SUBJECT_NAME_MAP = {v: k for k, v in DATASET_NAME_MAP.items()}
DATASET_ORDER = ["SciQ", "OpenBookQA", "ARC-Challenge", "RACE", "SQuAD v1.1"]
MODELS = ["Qwen 2.5 3B", "Google Gemini", "⚖️ Compare Both Side-by-Side"]

FIGURE_TITLES = {
    "01_overall_score_bars.png":         "Overall Score Comparison (Multi-Model)",
    "02_latency_vs_accuracy.png":        "Latency vs. Accuracy",
    "03_subject_heatmap.png":            "LLM Score Heatmap (Model × Subject)",
    "04_subject_bar_comparison.png":     "Per-Subject Bar Comparison",
    "05_exact_match_rate.png":           "Exact Match Rate Ranking",
    "06_subject_metric_heatmap.png":     "Subject Multi-Metric Heatmap",
    "07_score_distribution.png":         "Score Distribution per Model",
    "08_latency_distribution.png":       "Latency Distribution per Model",
    "09_tokens_vs_latency.png":          "Token Count vs. Latency",
    "10_qwen_vs_gemini_per_dataset.png": "Qwen vs. Gemini: Per-Dataset Comparison (5 Datasets)",
    "11_qwen_vs_gemini_scorecard.png":   "Qwen vs. Gemini: Executive Scorecard & Win Matrix",
}


@st.cache_data(ttl=30)
def load_leaderboard() -> pd.DataFrame | None:
    p = RESULTS_DIR / "leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    num = ["n_questions", "exact_match_rate", "avg_rouge_l", "avg_bert_score_f1",
           "avg_llm_score_1_5", "avg_llm_score_0_10", "avg_latency_s", "n_errors"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_subject_leaderboard() -> pd.DataFrame | None:
    p = RESULTS_DIR / "subject_leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["n_questions", "exact_match_rate", "avg_rouge_l", "avg_bert_score_f1",
              "avg_llm_score_1_5", "avg_llm_score_0_10"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_qwen_scored() -> pd.DataFrame | None:
    """Load Qwen per-question scored results from results/scored_results.csv."""
    p = RESULTS_DIR / "scored_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["exact_match", "rouge_l", "bert_score_f1", "llm_score_0_10", "llm_score_1_5",
              "latency_s", "prompt_tokens", "completion_tokens", "total_tokens"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_gemini_scored() -> pd.DataFrame | None:
    """Load Gemini per-question scored results from results/gemini_scored_results.csv."""
    p = RESULTS_DIR / "gemini_scored_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["exact_match", "token_f1", "token_precision", "token_recall",
              "rouge_l", "char_similarity", "contains_match"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_gemini_leaderboard() -> pd.DataFrame | None:
    """Load Gemini aggregated leaderboard from results/gemini_leaderboard.csv."""
    p = RESULTS_DIR / "gemini_leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["n_questions", "exact_match_rate", "avg_token_f1", "avg_rouge_l",
              "avg_char_similarity", "contains_match_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def missing_data_warning(label: str) -> None:
    st.warning(
        f"**{label}** not found. "
        "Run the evaluation script to generate it.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Global Session State & Synchronization
# ---------------------------------------------------------------------------
if "active_model" not in st.session_state:
    st.session_state["active_model"] = "Qwen 2.5 3B"
if "sb_model_key" not in st.session_state:
    st.session_state["sb_model_key"] = st.session_state["active_model"]
if "main_model_key" not in st.session_state:
    st.session_state["main_model_key"] = st.session_state["active_model"]

def on_sidebar_model_change():
    val = st.session_state["sb_model_key"]
    st.session_state["active_model"] = val
    st.session_state["main_model_key"] = val

def on_main_model_change():
    val = st.session_state["main_model_key"]
    st.session_state["active_model"] = val
    st.session_state["sb_model_key"] = val

# ---------------------------------------------------------------------------
# Sidebar Navigation & Model Selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 16px 0 10px;'>
            <span style='font-size:2.4rem;'>🎓</span><br>
            <span style='font-size:1.15rem; font-weight:800;
                         background: linear-gradient(90deg,#60a5fa,#818cf8);
                         -webkit-background-clip:text;
                         -webkit-text-fill-color:transparent;'>
                EduBench-Local
            </span><br>
            <span style='font-size:0.75rem; color:#4b6a9c;'>
                Multi-Model LLM Benchmark
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1e2a45; margin:4px 0 14px;'>", unsafe_allow_html=True)

    # ── Interactive Model Selection ──
    st.markdown("**🤖 Active Evaluation Model**")
    st.selectbox(
        "Model Selection",
        MODELS,
        key="sb_model_key",
        on_change=on_sidebar_model_change,
        label_visibility="collapsed",
        help="Select the model to inspect its detailed evaluation metrics and generated answers."
    )
    active_model = st.session_state["active_model"]

    if active_model == "Qwen 2.5 3B":
        st.caption("🔵 **Qwen 2.5 (3B)** · Local Ollama · 750 questions evaluated")
    elif active_model == "Google Gemini":
        st.caption("🟠 **Google Gemini** · Cloud Flash API · 5-question micro-sample")
    else:
        st.caption("⚖️ **Side-by-Side** · Head-to-head comparison across 5 datasets")

    st.markdown("<hr style='border-color:#1e2a45; margin:14px 0 16px;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊  Leaderboard & Analytics", "🖼️  Visualizations", "🧪  Live Model Playground"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2a45; margin:16px 0 12px;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem; color:#3a5278; text-align:center;'>"
        "EduBench-Local Evaluation Suite<br>"
        f"Data dir: <code style='color:#4b6a9c'>{RESULTS_DIR.name}/</code>"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 — Leaderboard & Analytics (Interactive Model Switching)
# ===========================================================================
if page == "📊  Leaderboard & Analytics":
    active_model = st.session_state["active_model"]

    st.markdown(
        f"""
        <div class='page-hero'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <h1>📊 Benchmark Leaderboard &amp; Analytics</h1>
                    <p>Evaluating educational QA performance across SciQ, OpenBookQA, ARC-Challenge, RACE, and SQuAD v1.1</p>
                </div>
                <div>
                    <span class='model-pill {"pill-qwen" if active_model == "Qwen 2.5 3B" else ("pill-gemini" if active_model == "Google Gemini" else "pill-qwen")}'>
                        {active_model}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Top interactive model switcher tabs ──
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.radio(
            "Select Model View",
            MODELS,
            horizontal=True,
            key="main_model_key",
            on_change=on_main_model_change,
        )

    model_view = st.session_state["active_model"]
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # VIEW 1: Qwen 2.5 3B
    # -----------------------------------------------------------------------
    if model_view == "Qwen 2.5 3B":
        lb = load_leaderboard()
        sub_lb = load_subject_leaderboard()
        qwen_scored = load_qwen_scored()

        if lb is None or lb.empty:
            missing_data_warning("results/leaderboard.csv")
            st.stop()

        qwen_row = lb.iloc[0]
        qwen_em = float(qwen_row["exact_match_rate"]) * 100
        qwen_rl = float(qwen_row.get("avg_rouge_l", 0.137))
        qwen_llm = float(qwen_row.get("avg_llm_score_1_5", 4.113))
        qwen_n = int(qwen_row["n_questions"])

        # ── 1. Top Metric Cards ──
        st.subheader("🔵 Qwen 2.5 (3B) Performance Spotlight")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🤖 Model Architecture", "Qwen 2.5 (3B)")
        c2.metric("✅ Exact Match Rate", f"{qwen_em:.1f}%")
        c3.metric("📈 Mean ROUGE-L", f"{qwen_rl:.3f}")
        c4.metric("🎯 Quality Score", f"{qwen_llm:.3f}", delta="LLM Score (1–5)", delta_color="off")
        c5.metric("📝 Questions Evaluated", f"{qwen_n:,}", delta="150 / dataset", delta_color="off")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 2. Per-Dataset Performance Breakdown ──
        if sub_lb is not None and not sub_lb.empty:
            st.subheader("Per-Dataset Performance Breakdown")
            col_l, col_r = st.columns([3, 2], gap="large")

            with col_l:
                disp_sub = sub_lb.copy()
                disp_sub["Dataset"] = disp_sub["subject"].map(DATASET_NAME_MAP).fillna(disp_sub["subject"])
                disp_sub["sort_order"] = disp_sub["Dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                disp_sub = disp_sub.sort_values("sort_order").drop(columns=["sort_order"])

                disp_sub["Questions (N)"] = disp_sub["n_questions"].astype(int)
                disp_sub["Exact Match %"] = (disp_sub["exact_match_rate"] * 100).map("{:.2f}%".format)
                disp_sub["ROUGE-L"] = disp_sub.get("avg_rouge_l", 0).map("{:.4f}".format)
                disp_sub["Semantic F1"] = disp_sub.get("avg_bert_score_f1", 0).map("{:.4f}".format)
                disp_sub["Quality Score"] = disp_sub.get("avg_llm_score_1_5", 0).map("{:.4f}".format)

                show_cols = ["Dataset", "subject", "Questions (N)", "Exact Match %", "ROUGE-L", "Semantic F1", "Quality Score"]
                st.dataframe(
                    disp_sub[show_cols].rename(columns={"subject": "Internal Subject"}),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_r:
                chart_sub = sub_lb.copy()
                chart_sub["Dataset"] = chart_sub["subject"].map(DATASET_NAME_MAP).fillna(chart_sub["subject"])
                chart_sub["sort_order"] = chart_sub["Dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                chart_sub = chart_sub.sort_values("sort_order")
                chart_df = chart_sub.set_index("Dataset")[["avg_rouge_l", "exact_match_rate"]]
                chart_df.columns = ["ROUGE-L", "Exact Match Rate"]
                st.markdown("**ROUGE-L vs. Exact Match by Dataset**")
                st.bar_chart(chart_df, height=270)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 3. Detailed Generated Answers Explorer ──
        if qwen_scored is not None and not qwen_scored.empty:
            st.subheader("Detailed Generated Answers Explorer (Qwen)")

            f1, f2 = st.columns([2, 2])
            with f1:
                sel_ds = st.selectbox("Filter by Dataset", ["All"] + DATASET_ORDER, key="qwen_ds_filter")
            with f2:
                search_query = st.text_input("🔍 Search in Questions / Answers", "", key="qwen_search")

            filtered = qwen_scored.copy()
            filtered["Dataset"] = filtered["subject"].map(DATASET_NAME_MAP).fillna(filtered["subject"])
            if sel_ds != "All":
                filtered = filtered[filtered["Dataset"] == sel_ds]
            if search_query.strip():
                q_low = search_query.strip().lower()
                filtered = filtered[
                    filtered["question"].str.lower().str.contains(q_low, na=False) |
                    filtered["student_answer"].str.lower().str.contains(q_low, na=False)
                ]

            st.caption(f"Showing **{len(filtered):,}** of **{len(qwen_scored):,}** answers")

            disp_cols = ["id", "Dataset", "subject", "question", "reference_answer", "student_answer",
                         "exact_match", "rouge_l", "llm_score_1_5", "latency_s"]
            avail_cols = [c for c in disp_cols if c in filtered.columns]
            st.dataframe(
                filtered[avail_cols].rename(columns={
                    "id": "ID", "Dataset": "Dataset", "subject": "Internal Subject", "question": "Question",
                    "reference_answer": "Ground Truth", "student_answer": "Model Answer",
                    "exact_match": "Exact Match", "rouge_l": "ROUGE-L", "llm_score_1_5": "Quality Score",
                    "latency_s": "Latency (s)"
                }),
                use_container_width=True,
                hide_index=True,
                height=380,
            )

            # Individual Question Deep-Dive Expanders
            st.markdown("#### 🔍 Individual Question Deep-Dive")
            deep_dive_samples = filtered.head(5)
            if len(filtered) > 5:
                st.caption(f"Displaying top {len(deep_dive_samples)} inspection cards from current filter")
            for _, row in deep_dive_samples.iterrows():
                em_status = "✅ Exact Match (1.0)" if row.get("exact_match") == 1 else "❌ Mismatch (0.0)"
                ds_name = row.get("Dataset") or DATASET_NAME_MAP.get(row.get("subject"), row.get("subject", "Item"))
                with st.expander(f"{ds_name} — {row['id']} [{em_status}]", expanded=False):
                    st.markdown(f"**Question**: {row['question']}")
                    st.markdown(f"**Ground Truth Reference**: `{row['reference_answer']}`")
                    st.markdown(f"**Qwen Answer**: `{row['student_answer']}`")
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric("Exact Match", int(row.get("exact_match", 0)))
                    m_c2.metric("ROUGE-L", f"{row.get('rouge_l', 0):.3f}")
                    m_c3.metric("Quality Score (LLM)", f"{row.get('llm_score_1_5', 0):.2f}")
                    lat = row.get("latency_s")
                    m_c4.metric("Latency", f"{lat:.2f}s" if pd.notna(lat) else "—")

    # -----------------------------------------------------------------------
    # VIEW 2: Google Gemini
    # -----------------------------------------------------------------------
    elif model_view == "Google Gemini":
        gem_scored = load_gemini_scored()
        gem_lb = load_gemini_leaderboard()

        if gem_scored is None or gem_scored.empty:
            missing_data_warning("results/gemini_scored_results.csv")
            st.info("Run `python gemini_evaluate.py` followed by `python gemini_evaluate_metrics.py` to generate Gemini scores.")
            st.stop()

        n_gem = len(gem_scored)
        gem_em = float(gem_scored["exact_match"].mean()) * 100
        gem_rl = float(gem_scored["rouge_l"].mean())
        gem_f1 = float(gem_scored["token_f1"].mean())

        # ── 1. Top Metric Cards ──
        st.subheader("🟠 Google Gemini Performance Spotlight")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🤖 Model Architecture", "Gemini (Flash)")
        c2.metric("✅ Exact Match Rate", f"{gem_em:.1f}%")
        c3.metric("📈 Mean ROUGE-L", f"{gem_rl:.3f}")
        c4.metric("🎯 Quality Score", f"{gem_f1:.3f}", delta="Token F1", delta_color="off")
        c5.metric("📝 Questions Evaluated", f"{n_gem}", delta="Micro-sample", delta_color="off")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 2. Per-Dataset Performance Breakdown ──
        if gem_lb is not None and not gem_lb.empty:
            st.subheader("Per-Dataset Performance Breakdown")
            col_gl, col_gr = st.columns([3, 2], gap="large")

            with col_gl:
                disp_glb = gem_lb.copy()
                disp_glb["sort_order"] = disp_glb["dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                disp_glb = disp_glb.sort_values("sort_order").drop(columns=["sort_order"])

                disp_glb["Questions (N)"] = disp_glb["n_questions"].astype(int)
                disp_glb["Exact Match %"] = (disp_glb["exact_match_rate"] * 100).map("{:.2f}%".format)
                disp_glb["ROUGE-L"] = disp_glb["avg_rouge_l"].map("{:.4f}".format)
                disp_glb["Semantic F1"] = disp_glb["avg_token_f1"].map("{:.4f}".format)
                disp_glb["Quality Score"] = disp_glb["avg_char_similarity"].map("{:.4f}".format)

                show_gcols = ["dataset", "subject", "Questions (N)", "Exact Match %", "ROUGE-L", "Semantic F1", "Quality Score"]
                st.dataframe(
                    disp_glb[show_gcols].rename(columns={"dataset": "Dataset", "subject": "Internal Subject"}),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_gr:
                chart_glb = gem_lb.copy()
                chart_glb["sort_order"] = chart_glb["dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                chart_glb = chart_glb.sort_values("sort_order")
                chart_gdf = chart_glb.set_index("dataset")[["avg_rouge_l", "exact_match_rate"]]
                chart_gdf.columns = ["ROUGE-L", "Exact Match Rate"]
                st.markdown("**ROUGE-L vs. Exact Match by Dataset**")
                st.bar_chart(chart_gdf, height=270)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 3. Detailed Generated Answers Explorer ──
        st.subheader("Detailed Generated Answers Explorer (Gemini)")

        fg1, fg2 = st.columns([2, 2])
        with fg1:
            sel_ds = st.selectbox("Filter by Dataset", ["All"] + DATASET_ORDER, key="gem_ds_filter")
        with fg2:
            gem_search = st.text_input("🔍 Search in Questions / Answers", "", key="gem_search")

        g_filtered = gem_scored.copy()
        g_filtered["Dataset"] = g_filtered["source_dataset"]
        if sel_ds != "All":
            g_filtered = g_filtered[g_filtered["Dataset"] == sel_ds]
        if gem_search.strip():
            g_low = gem_search.strip().lower()
            g_filtered = g_filtered[
                g_filtered["question"].str.lower().str.contains(g_low, na=False) |
                g_filtered["gemini_answer"].str.lower().str.contains(g_low, na=False)
            ]

        st.caption(f"Showing **{len(g_filtered):,}** of **{len(gem_scored):,}** answers")

        g_disp_cols = ["id", "Dataset", "subject", "question", "reference_answer", "gemini_answer",
                       "exact_match", "rouge_l", "token_f1", "model"]
        avail_gcols = [c for c in g_disp_cols if c in g_filtered.columns]

        st.dataframe(
            g_filtered[avail_gcols].rename(columns={
                "id": "ID", "Dataset": "Dataset", "subject": "Internal Subject", "question": "Question",
                "reference_answer": "Ground Truth", "gemini_answer": "Model Answer",
                "exact_match": "Exact Match", "rouge_l": "ROUGE-L", "token_f1": "Quality Score",
                "model": "Model Used"
            }),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

        # Individual Question Deep-Dive Expanders
        st.markdown("#### 🔍 Individual Question Deep-Dive")
        for _, row in g_filtered.iterrows():
            em_status = "✅ Exact Match (1.0)" if row["exact_match"] == 1 else "❌ Mismatch (0.0)"
            ds_name = row.get("Dataset") or row.get("source_dataset", "Item")
            with st.expander(f"{ds_name} — {row['id']} [{em_status}]", expanded=False):
                st.markdown(f"**Question**: {row['question']}")
                if row.get("context") and pd.notna(row["context"]):
                    st.caption(f"**Passage Context**: {str(row['context'])[:300]}...")
                st.markdown(f"**Ground Truth Reference**: `{row['reference_answer']}`")
                st.markdown(f"**Gemini Answer**: `{row['gemini_answer']}`")
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Exact Match", int(row.get("exact_match", 0)))
                m_c2.metric("ROUGE-L", f"{row.get('rouge_l', 0):.3f}")
                m_c3.metric("Quality Score (F1)", f"{row.get('token_f1', 0):.3f}")
                m_c4.metric("Char Similarity", f"{row.get('char_similarity', 0):.3f}")

    # -----------------------------------------------------------------------
    # VIEW 3: Side-by-Side Comparison
    # -----------------------------------------------------------------------
    else:
        lb = load_leaderboard()
        sub_lb = load_subject_leaderboard()
        gem_scored = load_gemini_scored()

        st.subheader("⚖️ Head-to-Head Comparative Overview")

        # Top comparative cards
        qwen_em = float(lb["exact_match_rate"].iloc[0])*100 if lb is not None and not lb.empty else 2.9
        qwen_rl = float(lb["avg_rouge_l"].iloc[0]) if lb is not None and not lb.empty else 0.137
        gem_em  = float(gem_scored["exact_match"].mean())*100 if gem_scored is not None else 40.0
        gem_rl  = float(gem_scored["rouge_l"].mean()) if gem_scored is not None else 0.533

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Exact Match Rate", f"Qwen: {qwen_em:.1f}%", f"Gemini: {gem_em:.1f}%", delta_color="normal")
        k2.metric("Mean ROUGE-L Overlap", f"Qwen: {qwen_rl:.3f}", f"Gemini: {gem_rl:.3f}", delta_color="normal")
        k3.metric("Deployment Mode", "Qwen: Local Ollama", "Gemini: Cloud Flash API")
        k4.metric("Inference Privacy", "Qwen: 100% On-Device", "Gemini: API Cloud")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 5 Datasets Comparison Table ──
        st.subheader("Dataset-by-Dataset Benchmark Performance Matrix")

        comp_rows = []
        for ds_name in DATASET_ORDER:
            s = SUBJECT_NAME_MAP.get(ds_name, ds_name)

            # Qwen metrics
            q_match = sub_lb.loc[sub_lb["subject"] == s] if sub_lb is not None else None
            q_em = float(q_match["exact_match_rate"].values[0])*100 if q_match is not None and not q_match.empty else 0.0
            q_rl = float(q_match["avg_rouge_l"].values[0]) if q_match is not None and not q_match.empty else 0.0

            # Gemini metrics
            if gem_scored is not None:
                g_match = gem_scored[gem_scored["subject"] == s]
                g_em = float(g_match["exact_match"].mean())*100 if not g_match.empty else 0.0
                g_rl = float(g_match["rouge_l"].mean()) if not g_match.empty else 0.0
            else:
                g_em, g_rl = 0.0, 0.0

            winner = "Gemini" if g_em > q_em or g_rl > q_rl else ("Qwen" if q_em > g_em or q_rl > g_rl else "Parity")

            comp_rows.append({
                "Dataset": ds_name,
                "Internal Subject": s,
                "Qwen EM %": f"{q_em:.1f}%",
                "Gemini EM %": f"{g_em:.1f}%",
                "Qwen ROUGE-L": f"{q_rl:.3f}",
                "Gemini ROUGE-L": f"{g_rl:.3f}",
                "Winner": f"★ {winner}"
            })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # Show embedded comparison figures if present
        f10 = FIGURES_DIR / "10_qwen_vs_gemini_per_dataset.png"
        f11 = FIGURES_DIR / "11_qwen_vs_gemini_scorecard.png"
        if f10.exists() or f11.exists():
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.subheader("Publication Comparative Visualizations")
            if f10.exists():
                st.image(str(f10), caption="Figure 10: Head-to-Head Comparison Across 5 Datasets", use_container_width=True)
            if f11.exists():
                st.image(str(f11), caption="Figure 11: Executive Scorecard & Win Matrix", use_container_width=True)


# ===========================================================================
# PAGE 2 — Visualizations Gallery
# ===========================================================================
elif page == "🖼️  Visualizations":

    st.markdown(
        """
        <div class='page-hero'>
            <h1>🖼️ Benchmark Visualizations Gallery</h1>
            <p>Comprehensive comparative performance charts generated in <code>figures/</code> · click to enlarge</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_files = sorted(FIGURES_DIR.glob("*.png"))

    if not fig_files:
        st.info("No figures found in `figures/`. Run `python generate_visualizations.py` to produce them.", icon="ℹ️")
        st.stop()

    # Filter controls
    f_c1, f_c2 = st.columns([2, 3])
    with f_c1:
        layout_mode = st.radio("Layout Mode", ["2-Column Grid", "Full Width"], horizontal=True)

    st.caption(f"Displaying **{len(fig_files)}** publication-grade charts")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    if layout_mode == "Full Width":
        for fp in fig_files:
            title = FIGURE_TITLES.get(fp.name, fp.stem.replace("_", " ").title())
            st.markdown(f"#### {title}")
            st.image(str(fp), use_container_width=True)
            st.caption(f"`{fp.name}` · {fp.stat().st_size / 1024:.1f} KB")
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    else:
        pairs = list(zip(fig_files[::2], fig_files[1::2]))
        if len(fig_files) % 2 == 1:
            pairs.append((fig_files[-1], None))

        for fl, fr in pairs:
            col_l, col_r = st.columns(2, gap="medium")
            with col_l:
                tl = FIGURE_TITLES.get(fl.name, fl.stem.replace("_", " ").title())
                st.markdown(f"**{tl}**")
                st.image(str(fl), use_container_width=True)
                st.caption(f"`{fl.name}` · {fl.stat().st_size / 1024:.1f} KB")

            if fr is not None:
                with col_r:
                    tr = FIGURE_TITLES.get(fr.name, fr.stem.replace("_", " ").title())
                    st.markdown(f"**{tr}**")
                    st.image(str(fr), use_container_width=True)
                    st.caption(f"`{fr.name}` · {fr.stat().st_size / 1024:.1f} KB")
            st.markdown("")


# ===========================================================================
# PAGE 3 — Live Model Playground (Supports Qwen & Gemini)
# ===========================================================================
elif page == "🧪  Live Model Playground":

    st.markdown(
        """
        <div class='page-hero'>
            <h1>🧪 Live Model Playground</h1>
            <p>Query local Ollama models (Qwen) or Google Gemini in real-time with latency measurement</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Try to import ollama ──
    try:
        import ollama as _ollama
        _ollama_available = True
    except ImportError:
        _ollama_available = False

    # ── Discover available models ──
    def get_ollama_models() -> list[str]:
        if not _ollama_available:
            return []
        try:
            resp = _ollama.list()
            return [m.model for m in resp.models] if resp.models else []
        except Exception:
            return []

    local_models = get_ollama_models()
    all_options = ["Qwen 2.5 (3B Local)", "Google Gemini (Flash)"]
    if local_models:
        all_options += [f"{m} (Ollama)" for m in local_models if m not in ("qwen2.5:3b", "qwen2.5:3b-instruct")]

    s1, s2, s3 = st.columns([2, 1, 1], gap="large")
    with s1:
        chosen_playground_model = st.selectbox("🤖 Choose Model", options=all_options)
    with s2:
        temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.0, 0.05)
    with s3:
        max_tokens = st.slider("📏 Max Output Tokens", 50, 500, 200, 25)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    subj_col, _ = st.columns([2, 3])
    with subj_col:
        subject = st.selectbox(
            "📚 Benchmark Domain",
            ["Science", "General Science", "Science Challenge", "Reading Comprehension", "General QA"],
        )

    question = st.text_area(
        "✏️ Enter Educational Question",
        height=120,
        placeholder="e.g. Which branch of biology studies animal behavior?\nWhen cold temperatures are produced in a chemical reaction, the reaction is known as...",
        key="pg_question_input",
    )

    run_col, clear_col, _ = st.columns([1, 1, 4])
    run_btn = run_col.button("▶  Run Query", use_container_width=True)
    clear_btn = clear_col.button("🗑  Clear", use_container_width=True)

    if clear_btn:
        for k in ["pg_ans", "pg_lat", "pg_mod"]:
            st.session_state.pop(k, None)
        st.rerun()

    if run_btn:
        q = question.strip()
        if not q:
            st.warning("Please enter a question to evaluate.", icon="⚠️")
        else:
            prompt = (
                f"You are an educational assessment assistant.\n"
                f"Subject: {subject}\n\n"
                f"Question: {q}\n\n"
                f"Answer in as few words as possible:\nAnswer:"
            )

            # ── Run Gemini ──
            if "Gemini" in chosen_playground_model:
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment.")
                else:
                    with st.spinner("Calling Google Gemini via GenAI SDK …"):
                        t0 = time.perf_counter()
                        try:
                            from google import genai
                            from google.genai import types
                            client = genai.Client(api_key=api_key)
                            resp = client.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=temperature,
                                    max_output_tokens=max_tokens,
                                ),
                            )
                            el = time.perf_counter() - t0
                            st.session_state["pg_ans"] = resp.text.strip() if resp and resp.text else "No text returned"
                            st.session_state["pg_lat"] = el
                            st.session_state["pg_mod"] = "gemini-3.8-flash"
                        except Exception as exc:
                            st.session_state["pg_ans"] = f"ERROR: {exc}"
                            st.session_state["pg_lat"] = None
                            st.session_state["pg_mod"] = "Gemini"

            # ── Run Qwen / Ollama ──
            else:
                if not _ollama_available:
                    st.error("Ollama package not available.")
                else:
                    ollama_model = "qwen2.5:3b" if "Qwen" in chosen_playground_model else chosen_playground_model.replace(" (Ollama)", "")
                    with st.spinner(f"Querying {ollama_model} via Ollama …"):
                        t0 = time.perf_counter()
                        try:
                            res = _ollama.chat(
                                model=ollama_model,
                                messages=[{"role": "user", "content": prompt}],
                                options={"temperature": temperature, "num_predict": max_tokens},
                            )
                            el = time.perf_counter() - t0
                            st.session_state["pg_ans"] = res["message"]["content"].strip()
                            st.session_state["pg_lat"] = el
                            st.session_state["pg_mod"] = ollama_model
                        except Exception as exc:
                            st.session_state["pg_ans"] = f"ERROR: {exc}"
                            st.session_state["pg_lat"] = None
                            st.session_state["pg_mod"] = ollama_model

    # Display playground output
    if "pg_ans" in st.session_state:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        r_ans = st.session_state["pg_ans"]
        r_lat = st.session_state.get("pg_lat")
        r_mod = st.session_state.get("pg_mod", chosen_playground_model)

        h1, h2, h3 = st.columns(3)
        h1.metric("Model Used", r_mod)
        h2.metric("Latency", f"{r_lat:.2f}s" if r_lat is not None else "—")
        h3.metric("Temperature", f"{temperature:.2f}")

        if r_ans.startswith("ERROR:"):
            st.markdown(f"<div class='error-box'>{r_ans}</div>", unsafe_allow_html=True)
        else:
            st.markdown("**Generated Response**")
            st.markdown(f"<div class='response-box'>{r_ans}</div>", unsafe_allow_html=True)
