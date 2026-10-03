"""
EduBench-Local: Prompt Builder
===============================
Constructs prompts for Mistral 7B based on question type (open-ended vs MCQ).
Uses fixed templates from config to ensure fair evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_prompt(item):
    """
    Build a prompt for a given standardized question item.
    
    Args:
        item: dict with keys: question, context, question_type, choices, answer_key
        
    Returns:
        str: The formatted prompt string
    """
    question = item["question"].strip()
    context = item.get("context", "").strip()
    question_type = item.get("question_type", "open_ended")
    
    # Build context block
    if context:
        context_block = f"Context: {context}\n\n"
    else:
        context_block = ""
    
    if question_type == "mcq":
        return _build_mcq_prompt(question, context_block, item.get("choices", []))
    else:
        return _build_open_ended_prompt(question, context_block)


def _build_open_ended_prompt(question, context_block):
    """Build prompt for open-ended questions (SciQ, SQuAD)."""
    return config.PROMPT_OPEN_ENDED.format(
        question=question,
        context_block=context_block,
    )


def _build_mcq_prompt(question, context_block, choices):
    """Build prompt for multiple-choice questions (OpenBookQA, ARC, RACE)."""
    if not choices:
        # Fallback to open-ended if no choices available
        return _build_open_ended_prompt(question, context_block)
    
    # Format choices
    choices_lines = []
    for choice in choices:
        if isinstance(choice, dict):
            label = choice.get("label", "?")
            text = choice.get("text", "")
        elif isinstance(choice, (list, tuple)) and len(choice) == 2:
            label, text = choice
        else:
            continue
        choices_lines.append(f"{label}) {text}")
    
    choices_block = "\n".join(choices_lines)
    
    return config.PROMPT_MCQ.format(
        question=question,
        context_block=context_block,
        choices_block=choices_block,
    )


if __name__ == "__main__":
    # Quick test with sample items
    test_open = {
        "question": "What is photosynthesis?",
        "context": "Plants use sunlight to convert CO2 and water into glucose.",
        "question_type": "open_ended",
        "choices": [],
    }
    
    test_mcq = {
        "question": "What is the primary source of energy for Earth's climate system?",
        "context": "",
        "question_type": "mcq",
        "choices": [
            {"label": "A", "text": "The Moon"},
            {"label": "B", "text": "The Sun"},
            {"label": "C", "text": "Earth's core"},
            {"label": "D", "text": "Cosmic rays"},
        ],
    }
    
    print("=" * 60)
    print("OPEN-ENDED PROMPT:")
    print("=" * 60)
    print(build_prompt(test_open))
    print()
    print("=" * 60)
    print("MCQ PROMPT:")
    print("=" * 60)
    print(build_prompt(test_mcq))
