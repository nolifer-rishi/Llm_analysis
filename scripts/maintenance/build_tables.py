import pandas as pd
import re

html_tables = ""
js_declarations = ""
js_hide_all = ""
js_show_logic = ""

models_map = {
    'gemma2:2b': ('gemma-table', 'gemma'),
    'openai/gpt-oss-120b': ('gpt-table', 'gpt'),
    'qwen2.5:1.5b': ('qwen1b-table', 'qwen1b'),
    'deepseek-r1:1.5b': ('deepseek-table', 'deepseek'),
    'llama3.2:3b': ('llama-table', 'llama'),
    'qwen2.5:3b': ('qwen3b-table', 'qwen3b'),
    'gemini-3.6-flash': ('gemini-table', 'gemini'),
}

# 1. Aditya
df_ad = pd.read_csv('c:/Users/Rishi/Desktop/llm/results_aditya/final_analysis/aggregation/subject_metrics.csv')
for model in ['gemma2:2b', 'openai/gpt-oss-120b', 'qwen2.5:1.5b']:
    m_data = df_ad[df_ad['model'] == model]
    tid, dropdown_val = models_map[model]
    
    html = f'<!-- {model} Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
    for i, row in m_data.sort_values('avg_rouge_l', ascending=False).reset_index().iterrows():
        html += f'<tr><td class="rank">{i+1}</td><td><strong>{row["subject"]}</strong></td><td>{row["avg_rouge_l"]:.3f}</td><td>{row["avg_bertscore_f1"]:.3f}</td><td class="highlight">{row["avg_judge_overall"]:.2f}</td></tr>\n'
    html += '</tbody></table>\n\n'
    html_tables += html

# 2. Apurba
df_ap = pd.read_csv('c:/Users/Rishi/Desktop/llm/qnaanalysis-main_apurba/results/scored_results.csv')
grouped = df_ap.groupby(['model', 'subject'])[['rouge_l', 'bertscore_f1', 'judge_score']].mean().reset_index()
for model in ['deepseek-r1:1.5b', 'llama3.2:3b']:
    m_data = grouped[grouped['model'] == model]
    tid, dropdown_val = models_map[model]
    
    html = f'<!-- {model} Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
    for i, row in m_data.sort_values('rouge_l', ascending=False).reset_index().iterrows():
        html += f'<tr><td class="rank">{i+1}</td><td><strong>{row["subject"]}</strong></td><td>{row["rouge_l"]:.3f}</td><td>{row["bertscore_f1"]:.3f}</td><td class="highlight">{row["judge_score"]:.2f}</td></tr>\n'
    html += '</tbody></table>\n\n'
    html_tables += html

# 3. Ankan - Qwen
df_an = pd.read_csv('c:/Users/Rishi/Desktop/llm/Edulab-main_ankan/results/subject_leaderboard.csv')
m_data = df_an[df_an['model'] == 'qwen2.5:3b']
tid, dropdown_val = models_map['qwen2.5:3b']
html = f'<!-- qwen2.5:3b Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
for i, row in m_data.sort_values('avg_rouge_l', ascending=False).reset_index().iterrows():
    html += f'<tr><td class="rank">{i+1}</td><td><strong>{row["subject"]}</strong></td><td>{row["avg_rouge_l"]:.3f}</td><td>{row["avg_bert_score_f1"]:.3f}</td><td class="highlight">{row["avg_llm_score_1_5"]:.2f}</td></tr>\n'
html += '</tbody></table>\n\n'
html_tables += html

# 4. Ankan - Gemini
df_gem = pd.read_csv('c:/Users/Rishi/Desktop/llm/Edulab-main_ankan/results/gemini_leaderboard.csv')
tid, dropdown_val = models_map['gemini-3.6-flash']
html = f'<!-- gemini-3.6-flash Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>Token F1</th><th>Exact Match</th></tr></thead>\n<tbody>\n'
for i, row in df_gem.sort_values('avg_rouge_l', ascending=False).reset_index().iterrows():
    html += f'<tr><td class="rank">{i+1}</td><td><strong>{row["dataset"]}</strong></td><td>{row["avg_rouge_l"]:.3f}</td><td>{row["avg_token_f1"]:.3f}</td><td class="highlight">{row["exact_match_rate"]:.2f}</td></tr>\n'
html += '</tbody></table>\n\n'
html_tables += html


# Update index.html
with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('<!-- Blank Placeholder Table -->', html_tables + '<!-- Blank Placeholder Table -->')

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

# Update script.js
with open('c:/Users/Rishi/Desktop/llm/docs/script.js', 'r', encoding='utf-8') as f:
    js_content = f.read()

# Replace variables block
var_block = """    const overallTable = document.getElementById('overall-table');
    const mistralTable = document.getElementById('mistral-table');
    const orcaTable = document.getElementById('orca-table');
    const blankTable = document.getElementById('blank-table');
    const tables = {
        'overall': overallTable,
        'mistral': mistralTable,
        'orca': orcaTable,
        'gemma': document.getElementById('gemma-table'),
        'gpt': document.getElementById('gpt-table'),
        'qwen1b': document.getElementById('qwen1b-table'),
        'qwen3b': document.getElementById('qwen3b-table'),
        'deepseek': document.getElementById('deepseek-table'),
        'llama': document.getElementById('llama-table'),
        'gemini': document.getElementById('gemini-table')
    };
    
    // Hide all
    Object.values(tables).forEach(t => { if(t) t.style.display = 'none'; });
    blankTable.style.display = 'none';
    
    // Show selected
    if (tables[selectedValue]) {
        tables[selectedValue].style.display = 'table';
    } else {
        blankTable.style.display = 'table';
    }
}
"""

js_content = re.sub(r'    const overallTable = document.getElementById\(\'overall-table\'\);.*?}$', var_block, js_content, flags=re.DOTALL | re.MULTILINE)

with open('c:/Users/Rishi/Desktop/llm/docs/script.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print("Tables generated successfully")
