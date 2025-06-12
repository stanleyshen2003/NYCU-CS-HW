import json
import pandas as pd

with open('pid_map_content.json', encoding='utf-8') as f:
    data = json.load(f)

processed_data = []


for key, value in data.items():
    for i, question in enumerate(value):
        question['answers'] = '或'.join(question['answers'])
        question['answers'] = question['question'] + ' ' + question['answers']
        question.pop('question')
        question = question['answers']
    value = [x['answers'] for x in value]
    value = 'Q: ' + ' Q: '.join(value).replace('\n', ' ').replace('\r', ' ')
    print(key, value)
    processed_data.append({'pid': key, 'content': value})
    
df = pd.DataFrame(processed_data)
df.to_csv('pid_map_content.csv', index=False)
