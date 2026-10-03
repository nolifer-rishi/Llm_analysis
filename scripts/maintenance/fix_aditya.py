import pandas as pd
import re

models_map = {
    'gemma2:2b': ('gemma-table', 'gemma'),
    'openai/gpt-oss-120b': ('gpt-table', 'gpt'),
    'qwen2.5:1.5b': ('qwen1b-table', 'qwen1b'),
}

html_tables = ""

df_ad = pd.read_csv('c:/Users/Rishi/Desktop/llm/results_aditya/final_analysis/aggregation/subject_metrics.csv')
for model in ['gemma2:2b', 'openai/gpt-oss-120b', 'qwen2.5:1.5b']:
    m_data = df_ad[df_ad['model'] == model]
    tid, dropdown_val = models_map[model]
    
    html = f'<!-- {model} Table -->\n<table id="{tid}" style="display: none;">\n<thead><tr><th>Rank</th><th>Dataset</th><th>ROUGE-L</th><th>BERTScore</th><th>Judge (1-5)</th></tr></thead>\n<tbody>\n'
    for i, row in m_data.sort_values('avg_rouge_l', ascending=False).reset_index().iterrows():
        html += f'<tr><td class="rank">{i+1}</td><td><strong>{row["subject"]}</strong></td><td>{row["avg_rouge_l"]:.3f}</td><td>{row["avg_bertscore_f1"]:.3f}</td><td class="highlight">{row["avg_judge_overall"]:.2f}</td></tr>\n'
    html += '</tbody></table>\n\n'
    html_tables += html

# Replace in index.html
with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'<!-- gemma2:2b Table -->.*?<!-- deepseek-r1:1.5b Table -->', html_tables + '<!-- deepseek-r1:1.5b Table -->', content, flags=re.DOTALL)

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Aditya tables generated successfully")
