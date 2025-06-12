import json

file_path = "answer1.json"
with open(file_path, 'r', encoding="utf-8") as f:
    raw_data = json.load(f)
    
data = raw_data['answers']

sorted_data = sorted(data, key=lambda x: x['qid'])

with open('answer1_sorted.json', 'w') as f:
    json.dump({"answers": sorted_data}, f, indent=4)