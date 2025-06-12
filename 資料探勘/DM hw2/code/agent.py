import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import BERT_Model
from dataset import Word_dataset
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



class Agent():
    def __init__(self, mode='train', save_root='model/', epochs=10, batch_size=16, model_root=None, pretrained_root='google-bert/bert-base-uncased', lr=1e-7):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_root = save_root
        self.model = BERT_Model(pretrained_root=pretrained_root).to(self.device)
        if model_root is not None:
            self.model.load_state_dict(torch.load(model_root))
        self.tokenizer_root = pretrained_root
        self.mode = mode
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.15)
        self.loss_fn = nn.CrossEntropyLoss()
        os.makedirs(save_root, exist_ok=True)
        
    def train(self):
        self.writer = SummaryWriter(self.save_root)
        
        for epoch in range(self.epochs):
            print('Epoch:', epoch)
            self.model.train()
            total_loss = 0
            total_count = 0
            
            dataset = Word_dataset(mode=self.mode, tokenizer_root=self.tokenizer_root)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            if epoch == 4:
                # fix bert model
                for param in self.model.model.parameters():
                    param.requires_grad = False
            for data, label in tqdm.tqdm(dataloader, ncols=100):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                label = label.to(self.device, dtype=torch.long)
                self.optimizer.zero_grad()
                output = self.model(input_ids, attention_mask)
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_count += 1
            self.scheduler.step()
            total_loss /= total_count
            print('Train Loss:', total_loss)
            self.writer.add_scalar('Train/Loss', total_loss, epoch)
            if epoch % 1 == 0:
                total_loss, macro_f1 = self.evaluate()
                self.writer.add_scalar('Val/Loss', total_loss, epoch)
                self.writer.add_scalar('Val/Macro_F1', macro_f1, epoch)
            if (epoch+1) % 5 == 0:
                # save the bert model
                torch.save(self.model.state_dict(), self.save_root+'epoch_'+str(epoch)+'.pth')
                
    def evaluate(self):
        self.model.eval()
        dataset = Word_dataset('val')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        total_loss = 0
        data_count = 0
        correct_count = 0
        all_output = []
        all_labels = []
        with torch.no_grad():
            for data, label in dataloader:
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                label = label.to(self.device, dtype=torch.long)
                output = self.model(input_ids, attention_mask).squeeze(-1)
                loss = self.loss_fn(output, label)
                output = output.cpu().numpy()
                label = label.cpu().numpy()
                output = output.astype(int)
                output = np.argmax(output, axis=1)
                all_output.extend(output.tolist())
                all_labels.extend(label.tolist())
                correct_count += (output == label).sum()
                total_loss += loss.item()
                data_count += 1
        total_loss /= data_count
        print('Val loss:', total_loss)
        print('Val Accuracy:', correct_count / dataset.__len__() )
        print(classification_report(all_labels, np.array(all_output)))
        json = classification_report(all_labels, np.array(all_output), output_dict=True)
        
        return total_loss, json["macro avg"]["f1-score"]
    
    def inference(self):
        self.model.eval()
        results = []
        dataset = Word_dataset('test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in tqdm.tqdm(dataloader, ncols=100):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                output = self.model(input_ids, attention_mask).argmax().item()
                results.append(output)
        return results