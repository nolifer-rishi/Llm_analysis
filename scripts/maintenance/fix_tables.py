import pandas as pd
import re

html_tables = ""

models_map = {
    'gemma2:2b': ('gemma-table', 'gemma'),
    'openai/gpt-oss-120b': ('gpt-table', 'gpt'),
    'qwen2.5:1.5b': ('qwen1b-table', 'qwen1b'),
    'deepseek-r1:1.5b': ('deepseek-table', 'deepseek'),
    'llama3.2:3b': ('llama-table', 'llama'),
    'qwen2.5:3b': ('qwen3b-table', 'qwen3b'),
    'gemini-3.6-flash': ('gemini-table', 'gemini'),
}

# Helper to map prefixes to dataset names
def get_dataset_name(qid):
    if qid.startswith('sciq'): return 'SciQ'
    if qid.startswith('race'): return 'RACE'
    if qid.startswith('arc'): return 'ARC'
    if qid.startswith('squad'): return 'SQuAD'
    if qid.startswith('obqa') or qid.startswith('openbook'): return 'OpenBookQA'
    return 'Unknown'

# 1. Aditya
df_metrics = pd.read_json('c:/Users/Rishi/Desktop/llm/results_aditya/combined_metrics.jsonl', lines=True)
df_judge = pd.read_json('c:/Users/Rishi/Desktop/llm/results_aditya/llm_judge_results.jsonl', lines=True)
df_ad = pd.merge(df_metrics, df_judge[['question_id', 'model', 'judge_overall']], on=['question_id', 'model'], how='inner')
df_ad['dataset'] = df_ad['question_id'].apply(get_dataset_name)

grouped_ad = df_ad.groupby(['model', 'dataset'])[['rougeL_f1', 'bertscore_f1', 'judge_overall']].mean().reset_index()

for model in ['gemma2:2b', 'openai/gpt-oss-120b', 'qwen2.5:1.5b']:
    m_data = grouped_ad[grouped_ad['model'] == model]
    tid, dropdown_val = models_map[model]
    
    html = f'<!-- {model} Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
    for i, row in m_data.sort_values('rougeL_f1', ascending=False).reset_index().iterrows():
        html += f'<tr><td class="rank">{i+1}</td><td><strong>{row["dataset"]}</strong></td><td>{row["rougeL_f1"]:.3f}</td><td>{row["bertscore_f1"]:.3f}</td><td class="highlight">{row["judge_overall"]:.2f}</td></tr>\n'
    html += '</tbody></table>\n\n'
    html_tables += html

# 2. Apurba
df_ap = pd.read_csv('c:/Users/Rishi/Desktop/llm/qnaanalysis-main_apurba/results/scored_results.csv')
grouped_ap = df_ap.groupby(['model', 'source_dataset'])[['rouge_l', 'bertscore_f1', 'judge_score']].mean().reset_index()
for model in ['deepseek-r1:1.5b', 'llama3.2:3b']:
    m_data = grouped_ap[grouped_ap['model'] == model]
    tid, dropdown_val = models_map[model]
    
    html = f'<!-- {model} Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
    for i, row in m_data.sort_values('rouge_l', ascending=False).reset_index().iterrows():
        # Rename ARC-Challenge to ARC
        dataset_name = "ARC" if "ARC" in row["source_dataset"] else row["source_dataset"]
        html += f'<tr><td class="rank">{i+1}</td><td><strong>{dataset_name}</strong></td><td>{row["rouge_l"]:.3f}</td><td>{row["bertscore_f1"]:.3f}</td><td class="highlight">{row["judge_score"]:.2f}</td></tr>\n'
    html += '</tbody></table>\n\n'
    html_tables += html

# 3. Ankan - Qwen
df_an = pd.read_csv('c:/Users/Rishi/Desktop/llm/Edulab-main_ankan/results/subject_leaderboard.csv')
m_data = df_an[df_an['model'] == 'qwen2.5:3b']
tid, dropdown_val = models_map['qwen2.5:3b']
html = f'<!-- qwen2.5:3b Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
dataset_map = {
    'reading_comprehension_squad': 'SQuAD',
    'science': 'SciQ',
    'science_challenge': 'ARC',
    'general_science': 'OpenBookQA',
    'reading_comprehension': 'RACE'
}
for i, row in m_data.sort_values('avg_rouge_l', ascending=False).reset_index().iterrows():
    d_name = dataset_map.get(row["subject"], row["subject"])
    html += f'<tr><td class="rank">{i+1}</td><td><strong>{d_name}</strong></td><td>{row["avg_rouge_l"]:.3f}</td><td>{row["avg_bert_score_f1"]:.3f}</td><td class="highlight">{row["avg_llm_score_1_5"]:.2f}</td></tr>\n'
html += '</tbody></table>\n\n'
html_tables += html

# 4. Ankan - Gemini
df_gem = pd.read_csv('c:/Users/Rishi/Desktop/llm/Edulab-main_ankan/results/gemini_leaderboard.csv')
tid, dropdown_val = models_map['gemini-3.6-flash']
html = f'<!-- gemini-3.6-flash Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>Token F1</th><th>Exact Match</th></tr></thead>\n<tbody>\n'
for i, row in df_gem.sort_values('avg_rouge_l', ascending=False).reset_index().iterrows():
    d_name = "ARC" if "ARC" in row["dataset"] else row["dataset"]
    html += f'<tr><td class="rank">{i+1}</td><td><strong>{d_name}</strong></td><td>{row["avg_rouge_l"]:.3f}</td><td>{row["avg_token_f1"]:.3f}</td><td class="highlight">{row["exact_match_rate"]:.2f}</td></tr>\n'
html += '</tbody></table>\n\n'
html_tables += html


# Replace in index.html
with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'<!-- gemma2:2b Table -->.*?<!-- Blank Placeholder Table -->', html_tables + '<!-- Blank Placeholder Table -->', content, flags=re.DOTALL)

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed tables generated successfully")
