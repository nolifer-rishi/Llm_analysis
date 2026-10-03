"""
EduBench-Local: ROUGE-L Metric
================================
Computes ROUGE-L F-measure between generated and reference answers.
"""

from rouge_score import rouge_scorer


# Initialize scorer once (reused across calls)
_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def compute_rouge_l(reference, generated):
    """
    Compute ROUGE-L F-measure for a single pair.
    
    Args:
        reference: str, the reference answer
        generated: str, the generated answer
        
    Returns:
        float: ROUGE-L F-measure (0.0 to 1.0)
    """
    if not reference or not generated:
        return 0.0
    
    scores = _scorer.score(reference, generated)
    return round(scores['rougeL'].fmeasure, 4)


def compute_rouge_l_batch(references, generated_list):
    """
    Compute ROUGE-L F-measure for a batch of pairs.
    
    Args:
        references: list of str
        generated_list: list of str
        
    Returns:
        list of float: ROUGE-L F-measure per pair
    """
    assert len(references) == len(generated_list), \
        f"Length mismatch: {len(references)} refs vs {len(generated_list)} generated"
    
    return [compute_rouge_l(ref, gen) for ref, gen in zip(references, generated_list)]


if __name__ == "__main__":
    # Quick test
    ref = "Photosynthesis is the process by which plants convert sunlight into energy."
    gen = "Plants use sunlight to produce glucose through a process called photosynthesis."
    
    score = compute_rouge_l(ref, gen)
    print(f"ROUGE-L F1: {score}")
