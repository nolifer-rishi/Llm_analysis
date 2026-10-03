"""
EduBench-Local: Main Pipeline Orchestrator
=============================================
Runs the full evaluation pipeline or individual steps.

Usage:
    python run_pipeline.py                    # Run all steps
    python run_pipeline.py --step setup       # Only dataset setup
    python run_pipeline.py --step generate    # Only answer generation
    python run_pipeline.py --step evaluate    # Only metric evaluation
    python run_pipeline.py --step aggregate   # Only aggregation + stats
    python run_pipeline.py --step visualize   # Only visualization
    python run_pipeline.py --step generate --dry-run  # Test with 1 q per dataset
    python run_pipeline.py --force            # Force recompute (ignore checkpoints)
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))
import config


def step_setup(force=False):
    """Step 1: Load and standardize all datasets."""
    print("\n" + "=" * 70)
    print("STEP 1: Dataset Setup")
    print("=" * 70)
    
    from src.dataset_loader import save_dataset_sample, load_dataset_sample
    
    if not force and config.DATASET_SAMPLE_FILE.exists():
        print(f"Dataset sample already exists at {config.DATASET_SAMPLE_FILE}")
        data = load_dataset_sample()
    else:
        data = save_dataset_sample()
    
    return data


def step_generate(questions=None, dry_run=False, force=False):
    """Step 2: Generate answers using Mistral 7B."""
    print("\n" + "=" * 70)
    print("STEP 2: Answer Generation")
    print("=" * 70)
    
    from src.dataset_loader import load_dataset_sample
    from src.answer_generator import generate_answers
    
    if questions is None:
        questions = load_dataset_sample()
    
    if dry_run:
        # Take 1 question per dataset for testing
        seen = set()
        test_qs = []
        for q in questions:
            if q["dataset"] not in seen:
                test_qs.append(q)
                seen.add(q["dataset"])
        questions = test_qs
        print(f"\n🧪 DRY RUN: Testing with {len(questions)} questions (1 per dataset)")
        
        # Use a separate checkpoint for dry runs
        checkpoint = config.RESULTS_DIR / "raw_answers_dryrun.json"
        answers = generate_answers(questions, checkpoint_path=checkpoint, force=True)
    else:
        answers = generate_answers(questions, force=force)
    
    return answers


def step_evaluate(force=False):
    """Step 3: Compute all 3 metrics."""
    print("\n" + "=" * 70)
    print("STEP 3: Metric Evaluation")
    print("=" * 70)
    
    from src.evaluator import evaluate_all
    
    df = evaluate_all(force=force)
    return df


def step_aggregate(df=None):
    """Step 4: Aggregate results and run statistical tests."""
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
    """
    Run the full pipeline or specific steps.
    
    Args:
        steps: list of step names to run, or None for all
        dry_run: if True, test with minimal data
        force: if True, recompute everything
    """
    all_steps = ["setup", "generate", "evaluate", "aggregate", "visualize"]
    
    if steps is None or "all" in steps:
        steps = all_steps
    
    print("+" + "=" * 68 + "+")
    print("|" + " EduBench-Local Pipeline ".center(68) + "|")
    print("|" + f" Model: {config.MODEL_DISPLAY_NAME} ".center(68) + "|")
    print("|" + f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print(f"\nSteps to run: {', '.join(steps)}")
    print(f"Dry run: {dry_run}")
    print(f"Force recompute: {force}")
    
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
        
        elapsed = time.time() - start_time
        
        print("\n" + "+" + "=" * 68 + "+")
        print("|" + " Pipeline Complete! ".center(68) + "|")
        print("|" + f" Total time: {elapsed/60:.1f} minutes ".center(68) + "|")
        print("+" + "=" * 68 + "+")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\nPipeline interrupted after {elapsed/60:.1f} minutes.")
        print("  Checkpoints have been saved. Re-run to resume from where you left off.")
        sys.exit(1)
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n\nPipeline failed after {elapsed/60:.1f} minutes: {e}")
        print("  Checkpoints have been saved. Fix the error and re-run to resume.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="EduBench-Local: Mistral 7B Educational QA Evaluation Pipeline"
    )
    parser.add_argument(
        "--step",
        nargs="+",
        choices=["all", "setup", "generate", "evaluate", "aggregate", "visualize"],
        default=["all"],
        help="Which pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with 1 question per dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute (ignore checkpoints)",
    )
    
    args = parser.parse_args()
    run_pipeline(steps=args.step, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
