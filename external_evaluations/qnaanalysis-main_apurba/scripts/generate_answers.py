"""
EduBench-Local — Multi-Model Answer Generation Engine
------------------------------------------------------
Loads standardized questions, applies the fixed tutor prompt template,
queries specified local LLMs via Ollama, measures latency & token counts,
and records answers incrementally to raw_answers.json.

Usage:
  python generate_answers.py [--models llama3.2:3b mistral:7b ...] [--dataset dataset_sample.json] [--output raw_answers.json] [--limit N]
"""

import os
import time
import json
import argparse
from datetime import datetime
import ollama

# Fixed Prompt Template from Section 5.2 of the Research Roadmap
PROMPT_TEMPLATE = """You are a helpful and knowledgeable tutor.
Answer the following question clearly and concisely, in 2-4 sentences.

Question: {question}
{context_block}Answer:"""


def format_prompt(question: str, context: str = "") -> str:
    context_clean = (context or "").strip()
    context_block = f"Context: {context_clean}\n" if context_clean else ""
    return PROMPT_TEMPLATE.format(question=question.strip(), context_block=context_block).strip()


def check_and_prepare_models(target_models: list[str]) -> list[str]:
    """Verify which requested models are installed in Ollama; pull if missing or prompt."""
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception as e:
        print(f"[!] Warning: Unable to query Ollama model list: {e}")
        return target_models

    ready_models = []
    for model in target_models:
        # Match with or without tag (e.g., llama3.2:3b or llama3.2:latest)
        base_name = model.split(":")[0]
        matching = [m for m in installed if m == model or m.startswith(f"{base_name}:")]
        if matching:
            ready_models.append(matching[0])
            print(f"[+] Model ready: {matching[0]}")
        else:
            print(f"[?] Model '{model}' is not currently pulled.")
            print(f"    To install: ollama pull {model}")
            # Try to pull if user wants or just flag it
            ready_models.append(model)
    return ready_models


def run_generation(dataset_path: str, models: list[str], output_path: str, limit: int = None):
    print(f"\n[*] Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit and limit > 0:
        dataset = dataset[:limit]
        print(f"[*] Limiting generation to first {limit} questions.")

    # Load existing raw answers if resuming
    existing_results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"[*] Found existing {len(existing_results)} answers in {output_path}. Resuming...")
        except Exception:
            existing_results = []

    # Map of (model, question_id) to skip duplicates
    completed_keys = {(r["model"], r["id"]) for r in existing_results}

    total_tasks = len(models) * len(dataset)
    current_count = len(existing_results)

    print(f"[*] Starting benchmark generation for {len(models)} model(s) across {len(dataset)} question(s)...")
    print(f"[*] Total generation tasks: {total_tasks} ({current_count} already completed)\n")

    for model_name in models:
        print(f"\n{'='*60}\n>>> Benchmarking Model: {model_name}\n{'='*60}")
        for i, item in enumerate(dataset):
            q_id = item["id"]
            if (model_name, q_id) in completed_keys:
                continue

            question = item["question"]
            context = item.get("context", "")
            ref_ans = item["reference_answer"]
            subject = item.get("subject", "general")
            prompt = format_prompt(question, context)

            print(f"[{i+1}/{len(dataset)}] [{subject.upper()}] {question[:65]}...")
            
            start_time = time.time()
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2}  # low temperature for reproducible QA
                )
                latency = round(time.time() - start_time, 3)
                generated_ans = response["message"]["content"].strip()
                
                # Approximate token count (words / 0.75 or response metadata if available)
                eval_count = response.get("eval_count")
                token_count = eval_count if eval_count else len(generated_ans.split())

                result_entry = {
                    "id": q_id,
                    "subject": subject,
                    "model": model_name,
                    "question": question,
                    "context": context,
                    "reference_answer": ref_ans,
                    "generated_answer": generated_ans,
                    "prompt": prompt,
                    "latency_sec": latency,
                    "token_count": token_count,
                    "timestamp": datetime.now().isoformat(),
                }

                existing_results.append(result_entry)
                completed_keys.add((model_name, q_id))

                # Incremental checkpoint save
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(existing_results, f, indent=2, ensure_ascii=False)

                print(f"    Done in {latency}s | Tokens: {token_count}")
                print(f"    Sample Ans: {generated_ans[:90]}...\n")

            except Exception as e:
                print(f"    [!] Error generating answer for {model_name} on {q_id}: {e}")
                time.sleep(1)

    print(f"\n[+] Generation completed! Saved {len(existing_results)} responses to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Answer Generation Engine for EduBench-Local")
    parser.add_argument("--dataset", type=str, default="dataset_sample.json", help="Path to input dataset JSON")
    parser.add_argument("--output", type=str, default="raw_answers.json", help="Path to save raw answers JSON")
    parser.add_argument("--models", nargs="+", default=["llama3.2:3b", "mistral:7b", "gemma2:2b", "phi3:mini", "qwen2.5:3b", "deepseek-r1:7b"],
                        help="List of Ollama models to benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process")
    args = parser.parse_args()

    models = check_and_prepare_models(args.models)
    run_generation(args.dataset, models, args.output, args.limit)


if __name__ == "__main__":
    main()
