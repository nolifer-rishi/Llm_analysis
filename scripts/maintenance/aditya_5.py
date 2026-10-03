import pandas as pd
import json
import re

html_tables = ""

models_map = {
    'gemma2:2b': ('gemma-table', 'gemma'),
    'openai/gpt-oss-120b': ('gpt-table', 'gpt'),
    'qwen2.5:1.5b': ('qwen1b-table', 'qwen1b'),
}

def get_dataset_name(qid):
    if qid.startswith('sciq'): return 'SciQ'
    if qid.startswith('race'): return 'RACE'
    if qid.startswith('arc'): return 'ARC'
    if qid.startswith('squad'): return 'SQuAD'
    if qid.startswith('obqa') or qid.startswith('openbook'): return 'OpenBookQA'
    return 'Unknown'

def robust_load(path):
    data = []
    import string
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        # split by '{"question_id"' to handle lines lacking newlines
        parts = content.split('{"question_id"')
        for p in parts:
            if not p: continue
            p = '{"question_id"' + p
            # truncate after the last matching '}'
            # A simple hack for flat dicts:
            try:
                # Find where it ends
                last_brace = p.rfind('}')
                if last_brace != -1:
                    clean_str = p[:last_brace+1]
                    data.append(json.loads(clean_str))
            except:
                pass
    return pd.DataFrame(data)

df_rouge = robust_load('c:/Users/Rishi/Desktop/llm/results_aditya/rouge_results.jsonl')
df_bert = robust_load('c:/Users/Rishi/Desktop/llm/results_aditya/bertscore_results.jsonl')
df_judge = robust_load('c:/Users/Rishi/Desktop/llm/results_aditya/llm_judge_results.jsonl')
df_comb = robust_load('c:/Users/Rishi/Desktop/llm/results_aditya/combined_metrics.jsonl')

df_merged = pd.concat([df_rouge, df_bert, df_judge, df_comb], ignore_index=True)

if not df_merged.empty and 'question_id' in df_merged.columns:
    df_merged['dataset'] = df_merged['question_id'].apply(get_dataset_name)
    
    # We want average of rougeL_f1, bertscore_f1, judge_overall
    metrics_cols = []
    if 'rougeL_f1' in df_merged.columns: metrics_cols.append('rougeL_f1')
    if 'bertscore_f1' in df_merged.columns: metrics_cols.append('bertscore_f1')
    if 'judge_overall' in df_merged.columns: metrics_cols.append('judge_overall')
    
    grouped = df_merged.groupby(['model', 'dataset'])[metrics_cols].mean().reset_index()

    for model in ['gemma2:2b', 'openai/gpt-oss-120b', 'qwen2.5:1.5b']:
        m_data = grouped[grouped['model'] == model]
        if m_data.empty: continue
        tid, dropdown_val = models_map[model]
        
        html = f'<!-- {model} Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
        
        sort_col = 'rougeL_f1' if 'rougeL_f1' in m_data.columns else 'dataset'
        for i, row in m_data.sort_values(sort_col, ascending=False).reset_index().iterrows():
            r_val = f'{row["rougeL_f1"]:.3f}' if 'rougeL_f1' in row and pd.notna(row["rougeL_f1"]) else 'N/A'
            b_val = f'{row["bertscore_f1"]:.3f}' if 'bertscore_f1' in row and pd.notna(row["bertscore_f1"]) else 'N/A'
            j_val = f'{row["judge_overall"]:.2f}' if 'judge_overall' in row and pd.notna(row["judge_overall"]) else 'N/A'
            html += f'<tr><td class="rank">{i+1}</td><td><strong>{row["dataset"]}</strong></td><td>{r_val}</td><td>{b_val}</td><td class="highlight">{j_val}</td></tr>\n'
        html += '</tbody></table>\n\n'
        html_tables += html

    with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the tables
    content = re.sub(r'<!-- gemma2:2b Table -->.*?<!-- deepseek-r1:1.5b Table -->', html_tables + '<!-- deepseek-r1:1.5b Table -->', content, flags=re.DOTALL)
    
    with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Done generating aditya 5 datasets.")
else:
    print("Failed to merge.")
