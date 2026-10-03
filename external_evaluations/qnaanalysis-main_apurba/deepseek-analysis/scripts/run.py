"""
DeepSeek-R1:1.5b | SciQ + RACE | 300-Question Benchmark
========================================================
150 SciQ + 150 RACE = 300 questions total.
Saves results to: deepseek-analysis/results/raw_answers_1.5b.json  (incremental)

Run:
  python deepseek-analysis/scripts/run.py
"""

import os, re, sys, time, json
from datetime import datetime
from collections import Counter
import ollama

MODEL        = "deepseek-r1:1.5b"
SOURCES      = {"SciQ", "RACE"}

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET      = os.path.join(ROOT, "data", "dataset_sample.json")
OUT_DIR      = os.path.join(ROOT, "deepseek-analysis", "results")
OUT_FILE     = os.path.join(OUT_DIR, "raw_answers_1.5b.json")

os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = """You are a helpful and knowledgeable tutor.
Answer the following question clearly and concisely, in 2-4 sentences.

Question: {question}
{ctx}Answer:"""

def make_prompt(q, ctx=""):
    ctx = ctx.strip()
    return PROMPT.format(question=q.strip(), ctx=f"Context: {ctx}\n" if ctx else "").strip()

def strip_think(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()

def log(msg):
    print(msg, flush=True)

def save(data):
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_existing():
    if os.path.exists(OUT_FILE):
        try:
            data = json.load(open(OUT_FILE, encoding="utf-8"))
            log(f"[*] Resuming — {len(data)} answers already saved.")
            return data
        except:
            pass
    return []

def main():
    log("\n" + "="*60)
    log("  DeepSeek-R1:1.5b  |  SciQ + RACE  |  300 Questions")
    log("="*60)

    # Check model
    try:
        models = [m.model for m in ollama.list().models]
        match  = [m for m in models if "deepseek-r1:1.5b" in m]
        if not match:
            log(f"[!] deepseek-r1:1.5b not found in ollama. Run: ollama pull {MODEL}")
            sys.exit(1)
        log(f"[+] Model: {match[0]}")
    except Exception as e:
        log(f"[!] Ollama error: {e}"); sys.exit(1)

    # Load + filter dataset
    all_data = json.load(open(DATASET, encoding="utf-8"))
    subset   = [q for q in all_data if q.get("source_dataset") in SOURCES]
    dist     = Counter(q["source_dataset"] for q in subset)
    log(f"[*] Dataset: { {k: v for k,v in sorted(dist.items())} }")
    log(f"[*] Total  : {len(subset)} questions")
    log(f"[*] Output : {OUT_FILE}\n")

    # Resume support
    results  = load_existing()
    done_ids = {r["id"] for r in results}
    pending  = [q for q in subset if q["id"] not in done_ids]
    total    = len(subset)
    offset   = len(results)
    log(f"[*] Pending: {len(pending)}  |  Done: {offset}\n")

    for i, item in enumerate(pending, 1):
        q_id    = item["id"]
        source  = item.get("source_dataset", "")
        subject = item.get("subject", "")
        q_text  = item["question"]
        ctx     = item.get("context", "")
        ref     = item["reference_answer"]
        prompt  = make_prompt(q_text, ctx)

        idx = offset + i
        log(f"[{idx:>3}/{total}] [{source}] {q_text[:65]}...")

        t0 = time.time()
        try:
            resp        = ollama.chat(model=MODEL,
                            messages=[{"role":"user","content":prompt}],
                            options={"temperature":0.2, "num_predict":512})
            latency     = round(time.time()-t0, 2)
            raw         = resp["message"]["content"].strip()
            clean       = strip_think(raw)
            tokens      = resp.get("eval_count") or len(clean.split())
            had_think   = "<think>" in raw

            results.append({
                "id": q_id, "source_dataset": source, "subject": subject,
                "model": MODEL, "question": q_text, "context": ctx,
                "reference_answer": ref, "generated_answer": clean,
                "raw_answer": raw, "had_think_block": had_think,
                "prompt": prompt, "latency_sec": latency,
                "token_count": tokens, "timestamp": datetime.now().isoformat()
            })
            done_ids.add(q_id)
            save(results)

            tag = " [think]" if had_think else ""
            log(f"       {latency}s | {tokens} tokens{tag}")
            log(f"       {clean[:80]}...\n")

        except Exception as e:
            log(f"       [!] ERROR: {e}\n")
            time.sleep(3)

    log("="*60)
    log(f"[+] DONE — {len(results)}/{total} answers saved to:")
    log(f"    {OUT_FILE}")
    think_n = sum(1 for r in results if r.get("had_think_block"))
    if results:
        avg_lat = sum(r["latency_sec"] for r in results) / len(results)
        avg_tok = sum(r["token_count"] for r in results) / len(results)
        log(f"[*] <think> blocks : {think_n}/{len(results)}")
        log(f"[*] Avg latency    : {avg_lat:.1f}s")
        log(f"[*] Avg tokens     : {avg_tok:.0f}")
    log("="*60 + "\n")

if __name__ == "__main__":
    main()
