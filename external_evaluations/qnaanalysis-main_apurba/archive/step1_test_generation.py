"""
EduBench-Local — Step 1 Test Script
------------------------------------
Purpose: Load a tiny sample (5 questions) from the SciQ dataset and generate
answers using ONE local model (llama3.2:3b) via Ollama. This confirms the
full pipeline works before scaling up to more questions and more models.

Run this from your activated virtual environment:
    python step1_test_generation.py
"""

import ollama
from datasets import load_dataset

# -------------------------------------------------------------------
# 1. Load a tiny sample from SciQ (just 5 questions for this first test)
# -------------------------------------------------------------------
print("Loading SciQ dataset sample...")
sciq = load_dataset("allenai/sciq", split="test")
sample = sciq.shuffle(seed=42).select(range(5))  # just 5 for now

# -------------------------------------------------------------------
# 2. Fixed prompt template (same one used across all models later)
# -------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a helpful and knowledgeable tutor.
Answer the following question clearly and concisely, in 2-4 sentences.

Question: {question}

Answer:"""

MODEL_NAME = "llama3.2:3b"

# -------------------------------------------------------------------
# 3. Generate an answer for each question using the local model
# -------------------------------------------------------------------
results = []

for i, item in enumerate(sample):
    question = item["question"]
    reference_answer = item["correct_answer"]

    prompt = PROMPT_TEMPLATE.format(question=question)

    print(f"\n[{i+1}/5] Asking model: {question}")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    generated_answer = response["message"]["content"].strip()

    print(f"Reference answer: {reference_answer}")
    print(f"Model answer: {generated_answer}")

    results.append({
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "model": MODEL_NAME,
    })

# -------------------------------------------------------------------
# 4. Save results to a file so you can inspect them
# -------------------------------------------------------------------
import json
with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nDone! Results saved to test_results.json")
