"""
EduBench-Local — Academic Research Leaderboard & Analytics Portal
-------------------------------------------------------------------
An interactive research platform for benchmarking offline, open-source Large 
Language Models (LLMs) on educational Question Answering tasks using 
Tri-Metric Evaluation (ROUGE-L, BERTScore F1, Local LLM-as-Judge 1-5).
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduBench-Local | LLM Educational QA Benchmark",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Global High-End Research Dark Mode CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

    /* Global Background and Typography */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0B1120 0%, #030712 100%);
        color: #F3F4F6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Research Header */
    .research-hero {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(14px);
    }
    .research-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .research-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #CBD5E1;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Dedicated LLM-as-Judge Deep-Dive Section */
    .judge-panel {
        background: linear-gradient(135deg, rgba(49, 36, 17, 0.3) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -10px rgba(245, 158, 11, 0.15);
    }
    .judge-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    .judge-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FBBF24;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0;
    }

    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-judge { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-bert { background: rgba(139, 92, 246, 0.2); color: #C4B5FD; border: 1px solid rgba(139, 92, 246, 0.4); }
    .badge-rouge { background: rgba(59, 130, 246, 0.2); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-combined { background: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.4); }

    /* Qualitative Inspection Cards */
    .qa-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.4rem;
    }
    .answer-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 1.1rem;
        margin-top: 0.8rem;
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Streamlit Components */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0px 0px;
        color: #94A3B8;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.92rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.25) !important;
        color: #A5B4FC !important;
        border-top: 2px solid #6366F1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_benchmark_data():
    scored_csv = "results/scored_results.csv" if os.path.exists("results/scored_results.csv") else "scored_results.csv"
    scored_json = "results/scored_results.json" if os.path.exists("results/scored_results.json") else "scored_results.json"
    dataset_json = "data/dataset_sample.json" if os.path.exists("data/dataset_sample.json") else "dataset_sample.json"

    df, raw_dataset = None, None

    if os.path.exists(scored_csv):
        df = pd.read_csv(scored_csv)
    elif os.path.exists(scored_json):
        with open(scored_json, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

    if df is not None:
        # Standardize source_dataset names
        if "source_dataset" not in df.columns or df["source_dataset"].isnull().any():
            def resolve_dataset(row):
                qid = str(row.get("id", "")).lower()
                if qid.startswith("sciq"): return "SciQ"
                if qid.startswith("arc"): return "ARC-Challenge"
                if qid.startswith("obqa") or qid.startswith("openbook"): return "OpenBookQA"
                if qid.startswith("race"): return "RACE"
                if qid.startswith("squad"): return "SQuAD"
                return "General"
            df["source_dataset"] = df.apply(resolve_dataset, axis=1)

        # Standardize normalized judge & tri-metric combined scores
        if "normalized_judge_score" not in df.columns:
            df["normalized_judge_score"] = df["judge_score"].apply(
                lambda j: max(0.0, min(1.0, (float(j) - 1.0) / 4.0)) if float(j) > 0 else 0.0
            )
        if "combined_score" not in df.columns:
            df["combined_score"] = ((df["rouge_l"] + df["bertscore_f1"] + df["normalized_judge_score"]) / 3.0).round(4)

    if os.path.exists(dataset_json):
        with open(dataset_json, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)

    return df, raw_dataset


def min_max_normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(1.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    leaderboard = df.groupby("model").agg(
        total_answers=("id", "count"),
        avg_judge=("judge_score", "mean"),
        std_judge=("judge_score", "std"),
        avg_norm_judge=("normalized_judge_score", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
        std_bertscore=("bertscore_f1", "std"),
        avg_rouge=("rouge_l", "mean"),
        std_rouge=("rouge_l", "std"),
        avg_combined=("combined_score", "mean"),
        avg_latency=("latency_sec", "mean")
    ).reset_index()

    leaderboard["rouge_norm"] = min_max_normalize(leaderboard["avg_rouge"])
    leaderboard["bertscore_norm"] = min_max_normalize(leaderboard["avg_bertscore"])
    leaderboard["judge_norm"] = min_max_normalize(leaderboard["avg_judge"])

    leaderboard["composite_rank_score"] = leaderboard[
        ["rouge_norm", "bertscore_norm", "judge_norm"]
    ].mean(axis=1)

    leaderboard = leaderboard.sort_values("avg_combined", ascending=False).reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1
    return leaderboard


def setup_dark_matplotlib():
    """Configures matplotlib and seaborn for academic publication dark styling."""
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#0F172A",
        "figure.facecolor": "#080D1A",
        "grid.color": "#1E293B",
        "grid.linestyle": "--",
        "text.color": "#F1F5F9",
        "axes.labelcolor": "#CBD5E1",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "axes.edgecolor": "#334155"
    })


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
def main():
    df, raw_dataset = load_benchmark_data()

    # Research Header Hero Banner
    st.markdown("""
    <div class="research-hero">
        <div class="research-title">🎓 EduBench-Local Benchmark Portal</div>
        <div class="research-subtitle">
            A Comparative Empirical Performance Analysis of Open-Source LLMs for Educational QA using Tri-Metric Evaluation 
            (ROUGE-L, Semantic BERTScore F1, and Local LLM-as-Judge 1–5 Rubric Grading).
        </div>
        <div class="tag-container">
            <span class="hero-tag">🔬 Empirical Research Suite</span>
            <span class="hero-tag">📦 750 Test Questions</span>
            <span class="hero-tag">🌐 5 Standardized Benchmark Datasets</span>
            <span class="hero-tag">⚡ 100% Offline Local Ollama Execution</span>
            <span class="hero-tag">⚖️ Strict Pedagogical Rubric</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Benchmark Controls")
        if st.button("🔄 Refresh / Reload Benchmark Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        if df is not None and len(df) > 0:
            all_models = sorted(df["model"].unique())
            selected_models = st.multiselect("🤖 Select LLM Models", all_models, default=all_models)

            all_sources = sorted(df["source_dataset"].unique())
            selected_sources = st.multiselect("📚 Filter Benchmark Datasets", all_sources, default=all_sources)

            st.markdown("---")
            st.markdown("#### 📖 Evaluation Methodology")
            st.markdown("""
            1. **⚖️ LLM-as-Judge (1–5)**: Pedagogical accuracy, factual correctness & completeness.
            2. **🧠 BERTScore (F1)**: Deep semantic similarity with `roberta-large`.
            3. **📝 ROUGE-L (F1)**: Lexical n-gram and LCS surface overlap.
            4. **🌟 Tri-Metric Combined**: Unified normalized composite score:
               $$\\text{Combined} = \\frac{\\text{ROUGE-L} + \\text{BERTScore} + \\frac{\\text{Judge}-1}{4}}{3}$$
            """)
            st.markdown("---")
            st.caption("EduBench-Local v1.2 | Zero-cost Offline Framework")

    if df is None or len(df) == 0:
        st.error("⚠️ Scored benchmark results not found. Please verify `results/scored_results.csv` exists.")
        return

    # Filtered Data
    filtered_df = df[df["model"].isin(selected_models)]
    if selected_sources:
        filtered_df = filtered_df[filtered_df["source_dataset"].isin(selected_sources)]

    if len(filtered_df) == 0:
        st.warning("⚠️ No data records match the active sidebar filters.")
        return

    leaderboard = compute_leaderboard(filtered_df)

    # Top KPI Metrics Cards (Sleek Dark Glassmorphism)
    top_model = leaderboard.iloc[0]
    avg_judge = filtered_df["judge_score"].mean()
    avg_bert = filtered_df["bertscore_f1"].mean()
    avg_rouge = filtered_df["rouge_l"].mean()
    avg_comb = filtered_df["combined_score"].mean()
    avg_lat = filtered_df["latency_sec"].mean()

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card" style="border-left: 4px solid #10B981;">
            <div class="kpi-title">🏆 Top Ranked Model</div>
            <div class="kpi-value">{top_model['model']}</div>
            <div class="kpi-sub" style="color:#34D399;">🌟 Combined: {top_model['avg_combined']:.3f} | Rank #{top_model['rank']}</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #F59E0B; background: linear-gradient(135deg, rgba(49, 36, 17, 0.4) 0%, rgba(15, 23, 42, 0.9) 100%);">
            <div class="kpi-title">⚖️ Local LLM-as-Judge Score</div>
            <div class="kpi-value" style="color:#FBBF24;">{avg_judge:.2f}<span style="font-size:1.1rem; color:#94A3B8;"> / 5.0</span></div>
            <div class="kpi-sub" style="color:#FCD34D;">Normalized: {((avg_judge-1)/4):.1%} Factual Quality</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #8B5CF6;">
            <div class="kpi-title">🧠 Semantic BERTScore</div>
            <div class="kpi-value">{avg_bert:.3f}</div>
            <div class="kpi-sub" style="color:#C084FC;">RoBERTa-large Token F1</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #3B82F6;">
            <div class="kpi-title">📝 Lexical ROUGE-L</div>
            <div class="kpi-value">{avg_rouge:.3f}</div>
            <div class="kpi-sub" style="color:#60A5FA;">LCS Surface Overlap F1</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #06B6D4;">
            <div class="kpi-title">⚡ Tri-Metric Combined</div>
            <div class="kpi-value">{avg_comb:.3f}</div>
            <div class="kpi-sub" style="color:#22D3EE;">Avg Latency: {avg_lat:.2f}s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Leaderboard & Performance",
        "📊 Visualizations & Heatmaps",
        "🔍 Qualitative Answer Inspector",
        "📈 Statistical Significance",
        "📄 Research Paper & BibTeX"
    ])

    # =========================================================
    # TAB 1: LEADERBOARD & PERFORMANCE
    # =========================================================
    with tab1:
        st.markdown("### 🏅 Overall Benchmark Leaderboard")
        display_cols = {
            "rank": "Rank",
            "model": "Model Name",
            "avg_combined": "🌟 Combined Score (0–1)",
            "avg_judge": "⚖️ LLM Judge (1–5)",
            "avg_norm_judge": "Norm Judge (0–1)",
            "avg_bertscore": "🧠 BERTScore (F1)",
            "avg_rouge": "📝 ROUGE-L (F1)",
            "avg_latency": "⏱️ Latency (s)",
            "total_answers": "Questions Evaluated"
        }
        table_df = leaderboard[[c for c in display_cols.keys() if c in leaderboard.columns]].rename(columns=display_cols)
        st.dataframe(
            table_df.style.format({
                "🌟 Combined Score (0–1)": "{:.4f}",
                "⚖️ LLM Judge (1–5)": "{:.2f}",
                "Norm Judge (0–1)": "{:.3f}",
                "🧠 BERTScore (F1)": "{:.4f}",
                "📝 ROUGE-L (F1)": "{:.4f}",
                "⏱️ Latency (s)": "{:.2f}s"
            }).background_gradient(subset=["🌟 Combined Score (0–1)", "⚖️ LLM Judge (1–5)", "🧠 BERTScore (F1)"], cmap="Blues"),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # DEDICATED LLM-AS-JUDGE QUALITY & RUBRIC DEEP-DIVE
        st.markdown("""
        <div class="judge-panel">
            <div class="judge-header">
                <div class="judge-title">
                    ⚖️ Local LLM-as-Judge Performance Deep-Dive
                </div>
                <span class="badge badge-judge">Pedagogical Rubric Grade: 1 (Lowest) to 5 (Flawless)</span>
            </div>
            <p style="color:#CBD5E1; font-size:0.95rem; margin-bottom:1.2rem; line-height:1.5;">
                To evaluate open-ended tutoring explanations where single-word exact match fails, a local <code>llama3.2:3b</code> 
                serves as an automated referee. It grades each generated explanation on factual correctness, evidence adherence, and completeness against the ground-truth reference.
            </p>
        </div>
        """, unsafe_allow_html=True)

        jcol1, jcol2 = st.columns([1.2, 1.8])

        with jcol1:
            st.markdown("#### 📊 LLM Judge Rating Distribution (1–5)")
            rating_counts = filtered_df["judge_score"].value_counts().reindex([5, 4, 3, 2, 1], fill_value=0)
            total_evals = len(filtered_df)

            labels = [
                "⭐⭐⭐⭐⭐ 5 / 5 (Comprehensive & Accurate)",
                "⭐⭐⭐⭐ 4 / 5 (Mostly Correct, Minor Omission)",
                "⭐⭐⭐ 3 / 5 (Partially Correct)",
                "⭐⭐ 2 / 5 (Major Factual Inaccuracies)",
                "⭐ 1 / 5 (Completely Incorrect / Irrelevant)"
            ]
            colors = ["#10B981", "#3B82F6", "#F59E0B", "#F97316", "#EF4444"]

            for i, score_val in enumerate([5, 4, 3, 2, 1]):
                cnt = int(rating_counts.get(score_val, 0))
                pct = (cnt / total_evals) * 100 if total_evals > 0 else 0
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; font-size:0.88rem; margin-top:0.45rem; color:#E2E8F0;">
                    <span>{labels[i]}</span>
                    <span style="font-weight:700; color:{colors[i]};">{cnt} ({pct:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(pct / 100.0)

        with jcol2:
            st.markdown("#### 🎯 LLM-as-Judge Quality by Dataset Source")
            setup_dark_matplotlib()
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ds_judge = filtered_df.groupby("source_dataset")["judge_score"].mean().reset_index().sort_values("judge_score", ascending=False)
            
            bars = ax.barh(ds_judge["source_dataset"], ds_judge["judge_score"], color="#F59E0B", edgecolor="#FBBF24", height=0.55, alpha=0.9)
            ax.set_xlim(0, 5.2)
            ax.set_xlabel("Average LLM-as-Judge Score (1–5)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Benchmark Dataset", fontsize=11, fontweight="bold")
            ax.axvline(x=4.0, color="#10B981", linestyle="--", alpha=0.7, label="Proficiency Threshold (4.0/5.0)")
            ax.legend(loc="lower right", facecolor="#0F172A", edgecolor="#334155", fontsize=9)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.08, bar.get_y() + bar.get_height()/2, f"{width:.2f} / 5.0",
                        ha="left", va="center", color="#F8FAFC", fontweight="bold", fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")

        # SECTION 3: Performance Breakdown by Benchmark Dataset Source
        st.subheader("📚 Detailed Performance by Dataset Source")
        st.markdown("Comprehensive metric breakdown across Science QA (**SciQ**, **ARC-Challenge**, **OpenBookQA**) and Reading Comprehension (**RACE**, **SQuAD v1.1**):")
        
        ds_summary = filtered_df.groupby("source_dataset").agg(
            total_questions=("id", "count"),
            avg_judge=("judge_score", "mean"),
            avg_norm_judge=("normalized_judge_score", "mean"),
            avg_bertscore=("bertscore_f1", "mean"),
            avg_rouge=("rouge_l", "mean"),
            avg_combined=("combined_score", "mean"),
            avg_latency=("latency_sec", "mean")
        ).reset_index().rename(columns={
            "source_dataset": "Benchmark Dataset",
            "total_questions": "Question Count",
            "avg_judge": "⚖️ LLM Judge (1–5)",
            "avg_norm_judge": "Norm Judge (0–1)",
            "avg_bertscore": "🧠 BERTScore (F1)",
            "avg_rouge": "📝 ROUGE-L (F1)",
            "avg_combined": "🌟 Combined Score",
            "avg_latency": "⏱️ Latency (s)"
        }).sort_values("🌟 Combined Score", ascending=False)

        st.dataframe(ds_summary.style.format({
            "⚖️ LLM Judge (1–5)": "{:.2f}",
            "Norm Judge (0–1)": "{:.3f}",
            "🧠 BERTScore (F1)": "{:.3f}",
            "📝 ROUGE-L (F1)": "{:.3f}",
            "🌟 Combined Score": "{:.3f}",
            "⏱️ Latency (s)": "{:.2f}s"
        }).background_gradient(subset=["⚖️ LLM Judge (1–5)", "🌟 Combined Score"], cmap="YlOrBr"),
        use_container_width=True, hide_index=True)

    # =========================================================
    # TAB 2: VISUALIZATIONS & HEATMAPS
    # =========================================================
    with tab2:
        st.subheader("📊 Comparative Performance Visualizations")
        setup_dark_matplotlib()
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.markdown("##### 🏆 Multi-Metric Profile Across Datasets")
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ds_metrics = filtered_df.groupby("source_dataset").agg(
                Judge_Norm=("normalized_judge_score", "mean"),
                BERTScore=("bertscore_f1", "mean"),
                ROUGE_L=("rouge_l", "mean"),
                Combined=("combined_score", "mean")
            ).reset_index()
            
            x = np.arange(len(ds_metrics))
            width = 0.2
            ax.bar(x - 1.5*width, ds_metrics["Judge_Norm"], width, label="Norm Judge", color="#F59E0B")
            ax.bar(x - 0.5*width, ds_metrics["BERTScore"], width, label="BERTScore", color="#8B5CF6")
            ax.bar(x + 0.5*width, ds_metrics["ROUGE_L"], width, label="ROUGE-L", color="#3B82F6")
            ax.bar(x + 1.5*width, ds_metrics["Combined"], width, label="Combined Score", color="#10B981")
            
            ax.set_xticks(x)
            ax.set_xticklabels(ds_metrics["source_dataset"], rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Normalized Metric Score (0–1)", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1.1)
            ax.legend(facecolor="#0F172A", edgecolor="#334155", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

        with vcol2:
            st.markdown("##### ⚡ Quality (LLM Judge) vs. Generation Latency")
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ds_eff = filtered_df.groupby("source_dataset").agg(
                judge_score=("judge_score", "mean"),
                latency_sec=("latency_sec", "mean"),
                count=("id", "count")
            ).reset_index()
            
            sns.scatterplot(
                data=ds_eff, 
                x="latency_sec", 
                y="judge_score", 
                hue="source_dataset", 
                s=240, 
                palette="tab10", 
                ax=ax,
                edgecolor="#FFFFFF",
                linewidth=1.5
            )
            for _, row in ds_eff.iterrows():
                ax.text(row["latency_sec"] + 0.015, row["judge_score"] + 0.015, row["source_dataset"], 
                        color="#F8FAFC", fontsize=9, fontweight="bold")
            
            ax.set_xlabel("Average Generation Latency (seconds)", fontsize=10, fontweight="bold")
            ax.set_ylabel("Avg LLM Judge Score (1–5)", fontsize=10, fontweight="bold")
            ax.set_ylim(3.8, 4.5)
            ax.legend(facecolor="#0F172A", edgecolor="#334155", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("##### 🗺️ Cross-Domain Evaluation Metric Correlation Matrix")
        matrix_df = filtered_df.groupby("source_dataset").agg(
            judge=("judge_score", "mean"),
            norm_judge=("normalized_judge_score", "mean"),
            bertscore=("bertscore_f1", "mean"),
            rouge=("rouge_l", "mean"),
            combined=("combined_score", "mean"),
            latency=("latency_sec", "mean")
        ).rename(columns={
            "judge": "LLM Judge (1–5)",
            "norm_judge": "Norm Judge (0–1)",
            "bertscore": "BERTScore (F1)",
            "rouge": "ROUGE-L (F1)",
            "combined": "Combined Score",
            "latency": "Latency (s)"
        })

        fig, ax = plt.subplots(figsize=(10, 4.2))
        sns.heatmap(matrix_df, annot=True, fmt=".3f", cmap="mako", ax=ax, linewidths=1.2, linecolor="#1E293B")
        ax.set_title("Dataset vs. Metric Matrix Heatmap", fontweight="bold", pad=12, color="#F8FAFC")
        ax.set_xlabel("Evaluation Metric", fontsize=10, fontweight="bold")
        ax.set_ylabel("Dataset Source", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    # =========================================================
    # TAB 3: QUALITATIVE INSPECTOR
    # =========================================================
    with tab3:
        st.subheader("🔍 Side-by-Side Model Answer Inspector")
        st.markdown("Select any question from the 750-item benchmark suite to inspect prompts, supporting context passages, ground-truth references, generated answers, and score breakdowns:")
        
        unique_questions = filtered_df[["id", "question", "source_dataset", "subject"]].drop_duplicates()
        selected_qid = st.selectbox(
            "Select Question to Inspect",
            options=unique_questions["id"].tolist(),
            format_func=lambda q_id: f"[{unique_questions[unique_questions['id']==q_id]['source_dataset'].values[0]}] {unique_questions[unique_questions['id']==q_id]['question'].values[0]}"
        )

        q_rows = filtered_df[filtered_df["id"] == selected_qid]
        first_row = q_rows.iloc[0]

        st.markdown(f"""
        <div class="qa-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <span class="badge" style="background:#3B82F6; color:#FFF;">{first_row.get('source_dataset', 'Dataset')}</span>
                <span style="font-size:0.85rem; color:#94A3B8;">ID: <code>{first_row['id']}</code> | Subject: <code>{first_row.get('subject', 'General')}</code></span>
            </div>
            <div style="font-size:1.15rem; font-weight:600; color:#F8FAFC; margin-bottom:0.6rem;">
                ❓ {first_row['question']}
            </div>
            <div style="font-size:0.95rem; color:#34D399; font-weight:600;">
                🎯 Ground Truth Reference: <span style="color:#A7F3D0; font-weight:500;">{first_row['reference_answer']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if first_row.get("context") and str(first_row.get("context")).strip():
            with st.expander("📖 Supporting Context / Passage Provided to Model"):
                st.write(first_row["context"])

        st.markdown("#### 🤖 Model Generated Explanations & Evaluated Scores")

        cols = st.columns(len(q_rows))
        for idx, (_, row) in enumerate(q_rows.iterrows()):
            with cols[idx]:
                judge_val = row.get('judge_score', 'N/A')
                bert_val = row.get('bertscore_f1', 0)
                rouge_val = row.get('rouge_l', 0)
                comb_val = row.get('combined_score', 0)
                lat_val = row.get('latency_sec', 0)
                tok_val = row.get('token_count', 0)

                st.markdown(f"""
                <div style="background:rgba(30, 41, 59, 0.7); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:1.2rem; box-shadow:0 8px 20px rgba(0,0,0,0.4);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                        <h4 style="margin:0; color:#60A5FA;">🤖 {row['model']}</h4>
                        <span class="badge badge-combined">🌟 Combined: {comb_val:.3f}</span>
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.8rem;">
                        <span class="badge badge-judge">⚖️ Judge: {judge_val}/5</span>
                        <span class="badge badge-bert">🧠 BERTScore: {bert_val:.3f}</span>
                        <span class="badge badge-rouge">📝 ROUGE-L: {rouge_val:.3f}</span>
                        <span class="badge" style="background:rgba(255,255,255,0.06); color:#94A3B8;">⏱️ {lat_val:.2f}s</span>
                        <span class="badge" style="background:rgba(255,255,255,0.06); color:#94A3B8;">🔢 {tok_val} tokens</span>
                    </div>
                    <div class="answer-box">
                        {row['generated_answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================
    # TAB 4: STATISTICAL TESTS
    # =========================================================
    with tab4:
        st.subheader("📈 Statistical Significance Testing (Wilcoxon Signed-Rank Test)")
        st.markdown("Evaluates whether observed score differentials between models achieve statistical significance ($p < 0.05$):")
        if len(selected_models) > 1:
            try:
                from scipy.stats import wilcoxon
                pivot = filtered_df.pivot_table(index="id", columns="model", values="judge_score")
                matrix = pd.DataFrame(np.ones((len(selected_models), len(selected_models))), index=selected_models, columns=selected_models)
                
                for m1 in selected_models:
                    for m2 in selected_models:
                        if m1 != m2 and m1 in pivot.columns and m2 in pivot.columns:
                            s1, s2 = pivot[m1].dropna(), pivot[m2].dropna()
                            common = s1.index.intersection(s2.index)
                            if len(common) > 2:
                                diff = s1.loc[common] - s2.loc[common]
                                if np.all(diff == 0):
                                    matrix.loc[m1, m2] = 1.0
                                else:
                                    try:
                                        _, p_val = wilcoxon(s1.loc[common], s2.loc[common])
                                        matrix.loc[m1, m2] = round(p_val, 4)
                                    except Exception:
                                        matrix.loc[m1, m2] = np.nan

                st.write("Pairwise p-values on **LLM Judge Score** ($p < 0.05$ indicates significant difference):")
                st.dataframe(matrix.style.format("{:.4f}").background_gradient(cmap="Reds_r"), use_container_width=True)
            except Exception as e:
                st.warning(f"Wilcoxon test unavailable: {e}")
        else:
            st.info("ℹ️ Single-model baseline active (`llama3.2:3b`). When additional open-source models (e.g., `mistral:7b`, `phi3:mini`, `gemma2:2b`, `qwen2.5:3b`, `deepseek-r1:7b`) are benchmarked, pairwise Wilcoxon significance matrices will automatically populate here.")

    # =========================================================
    # TAB 5: RESEARCH PAPER & CITATION
    # =========================================================
    with tab5:
        st.subheader("📄 Research Paper & Reproduction Artifacts")
        st.markdown("""
        **Title:** *EduBench-Local: A Comparative Performance Analysis of Open-Source LLMs on Educational Question Answering Using Multi-Metric Evaluation*  
        **Benchmark Suite:** SciQ (150), ARC-Challenge (150), OpenBookQA (150), RACE (150), SQuAD v1.1 (150) — **Total: 750 Questions**
        """)

        with st.expander("📝 View Research Paper Abstract"):
            st.write("""
            Large Language Models (LLMs) are increasingly deployed as personalized tutoring and educational study aids. 
            However, deploying closed-source commercial models incurs significant financial costs, privacy constraints, 
            and dependency on internet connectivity. In this work, we present **EduBench-Local**, a reproducible, cost-free 
            benchmark framework designed to evaluate offline open-source LLMs on educational Question Answering (QA). 
            We benchmark **Llama 3.2 (3B)** across a diverse 750-question cross-domain suite. Model answers are evaluated 
            using a tri-metric framework spanning lexical overlap (ROUGE-L), contextual semantic similarity (BERTScore), 
            and pedagogical correctness via a local LLM-as-Judge (1–5 scale).
            """)

        st.markdown("#### 📜 Citation (BibTeX)")
        bibtex = """@article{edubench_local_2026,
  title={EduBench-Local: A Comparative Performance Analysis of Open-Source LLMs on Educational Question Answering Using Multi-Metric Evaluation},
  author={EduBench Research Team},
  journal={Department of Computer Science Technical Report},
  year={2026},
  url={https://github.com/edubench-local/edubench}
}"""
        st.code(bibtex, language="bibtex")

        st.markdown("#### 📥 Download Scored Data")
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Scored Results (CSV)",
            data=csv_data,
            file_name="edubench_scored_results.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
