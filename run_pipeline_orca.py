"""
EduBench-Local: Pipeline Orchestrator for Orca Mini 3B
=====================================================
Mirrors run_pipeline.py but uses config_orca for Orca-specific settings.
Results are saved to results_orca/ directory.
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# IMPORTANT: Swap config module so all src/ modules use Orca settings
import config_orca as config
sys.modules["config"] = config


def step_setup(force=False):
    """Step 1: Load and standardize all datasets."""
    print("\n" + "=" * 70)
    print("STEP 1: Dataset Setup (Orca - 50 samples/dataset)")
    print("=" * 70)
    
    from src.dataset_loader import save_dataset_sample, load_dataset_sample
    
    if not force and config.DATASET_SAMPLE_FILE.exists():
        print(f"Dataset sample already exists at {config.DATASET_SAMPLE_FILE}")
        data = load_dataset_sample()
    else:
        data = save_dataset_sample()
    
    return data


def step_generate(questions=None, dry_run=False, force=False):
    """Step 2: Generate answers using Orca Mini 3B."""
    print("\n" + "=" * 70)
    print("STEP 2: Answer Generation (Orca Mini 3B)")
    print("=" * 70)
    
    from src.dataset_loader import load_dataset_sample
    from src.answer_generator import generate_answers
    
    if questions is None:
        questions = load_dataset_sample()
    
    if dry_run:
        seen = set()
        test_qs = []
        for q in questions:
            if q["dataset"] not in seen:
                test_qs.append(q)
                seen.add(q["dataset"])
        questions = test_qs
        print(f"\n  DRY RUN: Testing with {len(questions)} questions (1 per dataset)")
        
        checkpoint = config.RESULTS_DIR / "raw_answers_dryrun.json"
        answers = generate_answers(questions, checkpoint_path=checkpoint, force=True)
    else:
        answers = generate_answers(questions, force=force)
    
    return answers


def step_evaluate(force=False):
    """Step 3: Compute all metrics."""
    print("\n" + "=" * 70)
    print("STEP 3: Metric Evaluation")
    print("=" * 70)
    
    from src.evaluator import evaluate_all
    df = evaluate_all(force=force)
    return df


def step_aggregate(df=None):
    """Step 4: Aggregate results and run stats."""
    print("\n" + "=" * 70)
    print("STEP 4: Aggregation & Statistics")
    print("=" * 70)
    
    from src.aggregator import build_leaderboard
    from src.statistics import run_all_statistics
    
    leaderboard = build_leaderboard(df)
    stats = run_all_statistics(df)
    
    return leaderboard, stats


def step_visualize(df=None):
    """Step 5: Generate all figures."""
    print("\n" + "=" * 70)
    print("STEP 5: Visualization")
    print("=" * 70)
    
    from src.visualizer import generate_all_figures
    generate_all_figures(df)


def run_pipeline(steps=None, dry_run=False, force=False):
    all_steps = ["setup", "generate", "evaluate", "aggregate", "visualize"]
    if steps is None or "all" in steps:
        steps = all_steps
    
    print("+" + "=" * 68 + "+")
    print("|" + " EduBench-Local Pipeline (Orca Mini) ".center(68) + "|")
    print("|" + f" Model: {config.MODEL_DISPLAY_NAME} ".center(68) + "|")
    print("|" + f" Samples: {config.SAMPLE_SIZE_PER_DATASET} per dataset ".center(68) + "|")
    print("+" + "=" * 68 + "+")
    
    start_time = time.time()
    df = None
    
    try:
        if "setup" in steps:
            data = step_setup(force=force)
        
        if "generate" in steps:
            step_generate(dry_run=dry_run, force=force)
        
        if "evaluate" in steps:
            df = step_evaluate(force=force)
        
        if "aggregate" in steps:
            step_aggregate(df)
        
        if "visualize" in steps:
            step_visualize(df)
            
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", nargs="+", choices=["all", "setup", "generate", "evaluate", "aggregate", "visualize"], default=["all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(steps=args.step, dry_run=args.dry_run, force=args.force)

if __name__ == "__main__":
    main()
