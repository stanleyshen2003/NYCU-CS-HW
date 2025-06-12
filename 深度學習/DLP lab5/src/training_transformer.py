import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        if args.checkpoint_path:
            self.model.load_transformer_checkpoint(args.checkpoint_path)
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, data_loader):
        total_loss = 0
        for imgs in tqdm(data_loader):
            self.optim.zero_grad()
            x = imgs.to(args.device)
            logits, z_indeces = self.model(x)
            z_indeces = F.one_hot(z_indeces.view(-1), num_classes=logits.size(-1)).float()
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), z_indeces)
            total_loss += loss.item()
            loss.backward()
            self.optim.step()
        
        total_loss /= len(data_loader)
        print(f"Training Loss: {total_loss}")
        return total_loss

    @torch.no_grad()
    def eval_one_epoch(self, dataloader):
        total_loss = 0
        for imgs in tqdm(dataloader):
            x = imgs.to(args.device)
            logits, z_indeces = self.model(x)
            z_indeces = z_indeces.view(-1)
            logits = logits.view(-1, logits.size(-1))
            one_hot = F.one_hot(z_indeces, num_classes=logits.size(-1)).float()
            loss = self.loss_fn(logits, one_hot)
            total_loss += loss.item()
        
        total_loss /= len(dataloader)
        print(f"Validation Loss: {total_loss}")
        return total_loss

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.args.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, betas=self.args.betas)
        #optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 10, 40], gamma=0.3)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    torch.random.manual_seed(0)
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to checkpoint.') # './transformer_checkpoints/epoch_20.pt'
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for Adam.')
    parser.add_argument('--weight-decay', type=float, default=2e-3, help='Weight decay for Adam.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    writer = SummaryWriter()
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f"Epoch {epoch}, lr {train_transformer.scheduler.get_last_lr()[0]}")
        train_loss = train_transformer.train_one_epoch(train_loader)
        eval_loss = train_transformer.eval_one_epoch(val_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", eval_loss, epoch)
        if epoch % args.save_per_epoch == 0:
            print(f"Saving checkpoint at epoch {epoch}")
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/epoch_{epoch}.pt")
        train_transformer.scheduler.step()
            