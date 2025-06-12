import json
from transformers import AutoTokenizer
import random
import re
class Word_dataset():
    def __init__(self, mode='train', tokenizer_root='google-bert/bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root)
        self.mode = mode
        assert mode in ['train', 'val', 'test']
        if mode == 'test':
            self.path = 'data/test.json'
        elif mode == 'val':
            self.path = 'data/train.json'  
        else:
            self.path = 'data/train'+random.choice([''])+'.json' #'_ar', '_it', '_ko', '_la', '_ru', '_zh-cn', 
        print(self.path)
        with open(self.path, 'r') as f:
            
            self.data = json.load(f)
            
        if mode == 'train' and self.path == 'data/train.json':
            self.data = self.data[:int(len(self.data)*0.9)]
        elif mode == 'val':
            self.data = self.data[int(len(self.data)*0.9):]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.path == 'data/test.json' or self.path == 'data/train.json':
            data = ' Title: ' + item['title'] + '.'+ ' Comment: ' + item['text'] + '.' + str(item['helpful_vote'])+' helpful votes.'
        else:
            data = item['text']
        
        # replace '[[...]]' as ''
        data = re.sub(r'\[\[.*\]\]', '', data)
        
        data = data.replace('<br /><br />', ' ').replace('<br />', ' ')
        data = data.split('.')
        random.shuffle(data)
        data = '.'.join(data)
        output = self.tokenizer(data, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        output = {key: value.squeeze(0) for key, value in output.items()}
        if self.mode == 'test':
            return output
        label = item['rating']
        return output, label - 1