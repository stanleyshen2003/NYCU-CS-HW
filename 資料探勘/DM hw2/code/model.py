from torch import nn
from transformers import BertModel
import torch

class BERT_Model(nn.Module):
    def __init__(self, pretrained_root = 'google-bert/bert-base-uncased'):
        super(BERT_Model, self).__init__()
        self.model = BertModel.from_pretrained(pretrained_root)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.model.config.hidden_size, 5),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(512, 1)
        )
        # self.linear = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 5)
            
        # )
        
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state[:, 0, :]
        output = self.linear(output)
        return output