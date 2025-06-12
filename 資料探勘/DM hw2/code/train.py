import os
from agent import Agent
    

if __name__ == '__main__':
    model_root = None
    epochs = 10
    batch_size = 32
    lr = 2e-5
    pretrained_root = 'google-bert/bert-base-uncased'
    pretrained_text = pretrained_root.split('/')[0]
    save_root = f'logs/epochs_{epochs}_lr_{lr}_batch_{batch_size}_{pretrained_text}_fix_full_no_video/'
    
    os.makedirs(save_root, exist_ok=True)
    agent = Agent(model_root=model_root, save_root = save_root, epochs=epochs, batch_size=batch_size, lr=lr)
    agent.train()
