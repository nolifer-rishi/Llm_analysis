"""
Builds the focused 300-question SciQ x RACE cross-domain benchmark file.
"""
import json
import collections

with open('data/sciq_origin.json', 'r', encoding='utf-8') as f:
    sciq = json.load(f)
with open('data/race_reading_origin.json', 'r', encoding='utf-8') as f:
    race = json.load(f)

for item in sciq:
    item['domain'] = 'Science (SciQ)'
for item in race:
    item['domain'] = 'Language/Reading (RACE)'

combined = sciq + race
counts = collections.Counter(x['domain'] for x in combined)
print(f'Total: {len(combined)} questions')
print(dict(counts))

with open('data/sciq_race_benchmark.json', 'w', encoding='utf-8') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
print('Saved: data/sciq_race_benchmark.json')
