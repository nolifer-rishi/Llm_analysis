"""
EduBench-Local: BERTScore Metric
==================================
Computes BERTScore F1 for semantic similarity between generated and reference answers.
Processes in batches to manage memory on limited hardware.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

try:
    from bert_score import score as bert_score_fn
except ImportError:
    print("WARNING: bert-score package not installed. Run: pip install bert-score")
    bert_score_fn = None


def compute_bertscore_batch(references, generated_list, batch_size=None):
    """
    Compute BERTScore F1 for a batch of (reference, generated) pairs.
    
    Args:
        references: list of str
        generated_list: list of str
        batch_size: int, process in chunks of this size (default: from config)
        
    Returns:
        list of float: BERTScore F1 per pair (0.0 to 1.0)
    """
    if bert_score_fn is None:
        return [0.0] * len(references)
    
    assert len(references) == len(generated_list), \
        f"Length mismatch: {len(references)} refs vs {len(generated_list)} generated"
    
    batch_size = batch_size or config.BERTSCORE_BATCH_SIZE
    
    # Handle empty inputs
    if not references:
        return []
    
    # Replace empty strings with a placeholder to avoid BERTScore errors
    refs_clean = [r if r.strip() else "N/A" for r in references]
    gens_clean = [g if g.strip() else "N/A" for g in generated_list]
    
    all_f1 = []
    
    for i in range(0, len(refs_clean), batch_size):
        batch_refs = refs_clean[i:i + batch_size]
        batch_gens = gens_clean[i:i + batch_size]
        
        P, R, F1 = bert_score_fn(
            batch_gens,
            batch_refs,
            lang="en",
            verbose=False,
            device="cpu",  # Safe default; BERTScore uses its own model
        )
        
        all_f1.extend([round(f.item(), 4) for f in F1])
    
    return all_f1


def compute_bertscore_single(reference, generated):
    """
    Compute BERTScore F1 for a single pair.
    
    Args:
        reference: str
        generated: str
        
    Returns:
        float: BERTScore F1
    """
    scores = compute_bertscore_batch([reference], [generated])
    return scores[0] if scores else 0.0


if __name__ == "__main__":
    # Quick test
    ref = "Photosynthesis is the process by which plants convert sunlight into energy."
    gen = "Plants use sunlight to produce glucose through a process called photosynthesis."
    
    score = compute_bertscore_single(ref, gen)
    print(f"BERTScore F1: {score}")
