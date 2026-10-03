import re

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the model-selector div completely
content = re.sub(r'<div class="model-selector"[^>]*>.*?</div>\s*</div>\s*<div class="card glass table-wrapper">', '</div>\n', content, flags=re.DOTALL)

# Now we need to extract all tables inside the leaderboard section.
# Let's find the section
leaderboard_match = re.search(r'(<section id="leaderboard"[^>]*>.*?</section>)', content, re.DOTALL)
if not leaderboard_match:
    print("Could not find leaderboard section")
    exit()

leaderboard_html = leaderboard_match.group(1)

# Extract all tables
tables = re.findall(r'<table id="([^"]+)".*?>(.*?)</table>', leaderboard_html, re.DOTALL)

overall_table = None
model_tables = []

model_names = {
    'mistral-table': 'Mistral 7B',
    'orca-table': 'Orca Mini 3B',
    'gemma-table': 'Gemma 2 2B',
    'gpt-table': 'GPT-OSS 120B',
    'qwen1b-table': 'Qwen 2.5 1.5B',
    'qwen3b-table': 'Qwen 2.5 3B',
    'deepseek-table': 'DeepSeek-R1 1.5B',
    'llama-table': 'Llama 3.2 3B',
    'gemini-table': 'Gemini 3.6 Flash'
}

for t_id, t_content in tables:
    if t_id == 'overall-table':
        overall_table = f'<table id="{t_id}">{t_content}</table>'
    elif t_id == 'blank-table':
        pass # Discard blank table
    else:
        # Remove display:none from table tag if it exists (but we didn't capture the tag, we re-create it)
        clean_table = f'<table id="{t_id}">{t_content}</table>'
        model_tables.append((t_id, clean_table))

if not overall_table:
    print("Could not find overall table")
    exit()

# Build the new HTML structure
new_html = '''<section id="leaderboard" class="section">
    <div class="container">
        <h2 class="section-title" style="margin-bottom: 2rem;">Performance Leaderboard</h2>

        <div class="card glass table-wrapper" style="margin-bottom: 4rem;">
            <h3 style="margin-bottom: 1rem; color: var(--accent-primary); font-size: 1.5rem; text-align: center;">Overall Models Comparison</h3>
            ''' + overall_table + '''
        </div>

        <h3 class="section-title" style="font-size: 2.2rem; margin-bottom: 2rem;">Individual Model Breakdowns</h3>
        <div class="tables-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 2rem;">
'''

for t_id, t_html in model_tables:
    name = model_names.get(t_id, t_id)
    new_html += f'''            <div class="card glass table-wrapper" style="padding: 1.5rem;">
                <h4 style="margin-bottom: 1rem; text-align: center; font-size: 1.2rem; color: var(--text-main);">{name}</h4>
                {t_html}
            </div>
'''

new_html += '''        </div>
    </div>
</section>'''

# Replace the old section with the new section
final_content = content[:leaderboard_match.start()] + new_html + content[leaderboard_match.end():]

# Also remove the updateLeaderboard function from script.js since we don't need it
with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
    f.write(final_content)

print("Updated UI successfully!")
