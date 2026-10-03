import re

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to remove class="highlight" from all Judge columns.
# In the Overall table, the structure is:
# <td>0.290</td>
# <td>0.876</td>
# <td class="highlight">4.78</td>
# <td class="highlight">70.7</td>
#
# We want to change the first highlight to just a normal td, or remove it everywhere except the last column of the overall table.
# Actually, let's just do a regex replace on the table rows.

def process_row(match):
    row_html = match.group(0)
    # Find all <td> tags
    tds = re.findall(r'<td[^>]*>.*?</td>', row_html, re.DOTALL)
    
    if len(tds) == 6:
        # Overall table: Rank, Model, ROUGE, BERT, Judge, Combined
        # 0: rank
        # 1: model
        # 2: ROUGE
        # 3: BERT
        # 4: Judge (remove highlight)
        # 5: Combined (keep highlight)
        tds[4] = re.sub(r'class="highlight"', '', tds[4]).replace('<td >', '<td>')
        
    elif len(tds) == 5:
        # Individual table: Rank, Dataset, ROUGE, BERT, Judge
        # 4: Judge (remove highlight)
        tds[4] = re.sub(r'class="highlight"', '', tds[4]).replace('<td >', '<td>')

    # Reconstruct the row
    # The original row might have whitespace, we can just join them and wrap in <tr>
    new_row = "<tr>\n"
    for td in tds:
        new_row += f"    {td}\n"
    new_row += "</tr>"
    return new_row

# The easiest way is to use Beautiful Soup, but since we might not have it, let's just use regex safely.
# Find all <tr>...</tr> blocks inside <tbody> tags.

tbodies = re.split(r'(<tbody>.*?</tbody>)', content, flags=re.DOTALL)
new_content = ""

for chunk in tbodies:
    if chunk.startswith('<tbody>'):
        # process rows inside this tbody
        # but wait, the chunk contains the whole tbody.
        def replace_tr(m):
            tr_content = m.group(0)
            tds = re.findall(r'<td([^>]*)>(.*?)</td>', tr_content, re.DOTALL)
            if len(tds) == 6:
                # 4th index is judge
                attrs, text = tds[4]
                attrs = attrs.replace('class="highlight"', '').replace('class=\'highlight\'', '').strip()
                if attrs:
                    tds[4] = (f' {attrs}', text)
                else:
                    tds[4] = ('', text)
            elif len(tds) == 5:
                # 4th index is judge
                attrs, text = tds[4]
                attrs = attrs.replace('class="highlight"', '').replace('class=\'highlight\'', '').strip()
                if attrs:
                    tds[4] = (f' {attrs}', text)
                else:
                    tds[4] = ('', text)
            elif len(tds) == 1 and 'pending' in tds[0][1]:
                return tr_content
                
            new_tr = "<tr>\n"
            for attrs, text in tds:
                new_tr += f"                                <td{attrs}>{text}</td>\n"
            new_tr += "                            </tr>"
            return new_tr
            
        new_chunk = re.sub(r'<tr>.*?</tr>', replace_tr, chunk, flags=re.DOTALL)
        new_content += new_chunk
    else:
        new_content += chunk

# Also bump the v=4 to v=5 for good measure
new_content = new_content.replace('styles.css?v=4', 'styles.css?v=5')

with open('c:/Users/Rishi/Desktop/llm/docs/index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)
    
print("Fixed highlights!")
