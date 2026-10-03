import pandas as pd
import re

# Data for the 9 models
models = [
    {"name": "Gemma 2 2B", "rouge": "0.290", "bert": "0.876", "judge": "4.78"},
    {"name": "GPT-OSS 120B", "rouge": "0.388", "bert": "0.872", "judge": "3.62"},
    {"name": "Qwen 2.5 1.5B", "rouge": "0.164", "bert": "0.860", "judge": "4.82"},
    {"name": "Mistral 7B", "rouge": "0.131", "bert": "0.852", "judge": "4.52"},
    {"name": "Qwen 2.5 3B", "rouge": "0.137", "bert": "0.863", "judge": "4.11"},
    {"name": "DeepSeek-R1 1.5B", "rouge": "0.100", "bert": "0.850", "judge": "3.99"},
    {"name": "Llama 3.2 3B", "rouge": "0.084", "bert": "0.840", "judge": "4.09"},
    {"name": "Orca Mini 3B", "rouge": "0.089", "bert": "0.844", "judge": "2.30"},
    {"name": "Gemini 3.6 Flash", "rouge": "0.533", "bert": "N/A", "judge": "N/A"},
]

# Generate HTML for overall table
tbody_html = ""
for i, m in enumerate(models):
    tbody_html += f'''                            <tr>
                                <td class="rank">{i+1}</td>
                                <td><strong>{m["name"]}</strong></td>
                                <td>{m["rouge"]}</td>
                                <td>{m["bert"]}</td>
                                <td class="highlight">{m["judge"]}</td>
                                <td class="highlight">-</td>
                            </tr>\n'''

# Generate HTML for dropdown
dropdown_html = '''                            <option value="overall" style="color: #000;">Overall Models Comparison</option>
                            <option value="mistral" style="color: #000;">Mistral 7B Breakdown</option>
                            <option value="orca" style="color: #000;">Orca Mini 3B Breakdown</option>
                            <option value="gemma" style="color: #000;">Gemma 2 2B Breakdown</option>
                            <option value="gpt" style="color: #000;">GPT-OSS 120B Breakdown</option>
                            <option value="qwen1b" style="color: #000;">Qwen 2.5 1.5B Breakdown</option>
                            <option value="qwen3b" style="color: #000;">Qwen 2.5 3B Breakdown</option>
                            <option value="deepseek" style="color: #000;">DeepSeek-R1 1.5B Breakdown</option>
                            <option value="llama" style="color: #000;">Llama 3.2 3B Breakdown</option>
                            <option value="gemini" style="color: #000;">Gemini 3.6 Flash Breakdown</option>
'''

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace dropdown
content = re.sub(
    r'<select id="model-select"[^>]*>.*?</select>',
    '<select id="model-select" onchange="updateLeaderboard()" style="background: transparent; color: inherit; border: none; font-family: inherit; font-size: 1rem; cursor: pointer; outline: none;">\n' + dropdown_html + '                        </select>',
    content,
    flags=re.DOTALL
)

# Replace overall table body
content = re.sub(
    r'<table id="overall-table">.*?<tbody>.*?</tbody>',
    '<table id="overall-table">\n                        <thead>\n                            <tr>\n                                <th>Rank</th>\n                                <th>Model</th>\n                                <th>ROUGE-L</th>\n                                <th>BERTScore</th>\n                                <th>Judge (1-5)</th>\n                                <th>Combined Score</th>\n                            </tr>\n                        </thead>\n                        <tbody>\n' + tbody_html + '                        </tbody>',
    content,
    flags=re.DOTALL
)

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated index.html")
