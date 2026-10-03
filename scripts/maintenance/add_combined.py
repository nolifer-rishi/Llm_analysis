import re

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

table_match = re.search(r'<table id="overall-table">.*?<tbody>(.*?)</tbody>', content, re.DOTALL)
if not table_match:
    print("Table not found")
    exit()

tbody = table_match.group(1)
rows = re.findall(r'<tr>(.*?)</tr>', tbody, re.DOTALL)

data = []
for row in rows:
    cols = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
    if len(cols) < 5: continue
    model_str = cols[1].strip()
    rouge = cols[2].strip()
    bert = cols[3].strip()
    judge_html = cols[4].strip()
    
    judge = re.sub(r'<[^>]+>', '', judge_html)
    
    if rouge == 'N/A' or bert == 'N/A' or judge == 'N/A':
        combined = -1
        combined_str = "N/A"
    else:
        try:
            r = float(rouge)
            b = float(bert)
            j = float(judge)
            combined = (r + b + (j / 5.0)) / 3.0 * 100.0
            combined_str = f"{combined:.1f}"
        except Exception as e:
            print(f"Error parsing {model_str}: {e}")
            combined = -1
            combined_str = "N/A"
            
    data.append({
        'model_html': model_str,
        'rouge': rouge,
        'bert': bert,
        'judge_html': judge_html,
        'judge_val': judge,
        'combined': combined,
        'combined_str': combined_str
    })

data.sort(key=lambda x: x['combined'], reverse=True)

new_tbody = "\n"
for i, d in enumerate(data):
    # Strip existing strong tags just in case we double wrap them
    clean_model = re.sub(r'<[^>]+>', '', d["model_html"])
    clean_judge = re.sub(r'<[^>]+>', '', d["judge_html"])
    new_tbody += f'''                            <tr>
                                <td class="rank">{i+1}</td>
                                <td><strong>{clean_model}</strong></td>
                                <td>{d["rouge"]}</td>
                                <td>{d["bert"]}</td>
                                <td class="highlight">{clean_judge}</td>
                                <td class="highlight">{d["combined_str"]}</td>
                            </tr>\n'''

new_table = f'<table id="overall-table">\n                        <thead>\n                            <tr>\n                                <th>Rank</th>\n                                <th>Model</th>\n                                <th>ROUGE-L</th>\n                                <th>BERTScore</th>\n                                <th>Judge (1-5)</th>\n                                <th>Combined Score</th>\n                            </tr>\n                        </thead>\n                        <tbody>{new_tbody}                        </tbody>\n                    </table>'

new_content = re.sub(r'<table id="overall-table">.*?</table>', new_table, content, flags=re.DOTALL)

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Added combined scores successfully!")
