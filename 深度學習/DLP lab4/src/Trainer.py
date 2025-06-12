import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from skimage.exposure import match_histograms
from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
#import imageio
import random
import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        self.beta = 0.0
        
    def update(self):
        # TODO
        self.current_epoch += 1
        if self.kl_anneal_type == 'Cyclical':
            self.beta = self.frange_cycle_linear(start=0.0, stop=1.0)
        elif self.kl_anneal_type == 'Monotonic':
            self.beta = min(1.0, self.beta + self.kl_anneal_ratio * self.current_epoch / self.kl_anneal_cycle)
        else:
            raise NotImplementedError
    
    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, start=0.0, stop=1.0):
        # TODO
        return min(1.0, start + (stop - start) * (self.current_epoch % self.kl_anneal_cycle) / self.kl_anneal_cycle * self.kl_anneal_ratio)

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.writer = SummaryWriter(log_dir=self.args.save_root)
  
    def forward(self, img_last, img, label, mode=None):
        # TODO
        img = self.frame_transformation(img)
        label = self.label_transformation(label)
        z, mu, logvar = self.Gaussian_Predictor(img, label)
        if mode == 'test':
            z = torch.randn_like(z)
        img_last = self.frame_transformation(img_last)
        fused = self.Decoder_Fusion(img_last, label, z)
        out = self.Generator(fused)
        return out, mu, logvar
    
    def store_parameters(self):
        # save args
        with open(os.path.join(self.args.save_root, 'args.yaml'), 'w') as f:
            for k, v in vars(self.args).items():
                f.write(f"{k}: {v}\n")
        
    def training_stage(self):
        self.store_parameters()
        for _ in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            total_loss = 0
            for (img, label) in (pbar := tqdm(train_loader, ncols=160)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                total_loss += loss
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.5f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.5f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            print(f'total loss: {total_loss}')
            self.writer.add_scalar('Loss/train', total_loss, self.current_epoch)
            if (self.current_epoch + 1) % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, 'checkpoint', f"epoch{self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            images, loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            self.make_gif(images, os.path.join(self.args.save_root, 'GIF',  f"epoch{self.current_epoch}.gif"))
        self.writer.add_scalar('Loss/validate', loss, self.current_epoch)
        print(f'validate loss: {loss}')

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        loss = 0
        self.optim.zero_grad()
        last = img[:, 0]
        for i in range(1, self.train_vi_len):
            if adapt_TeacherForcing:
                out, mu, logvar = self.forward(last, img[:, i], label[:, i])
                last = img[:, i].detach()
            else:
                out, mu, logvar = self.forward(last, img[:, i], label[:, i])
                # temp = out.detach()
                # last = [self.histogram_specification(temp[i].unsqueeze(0), last[i].unsqueeze(0)) for i in range(temp.shape[0])]
                # last = torch.cat(last, axis=0)
                last = out.detach()
            mse = self.mse_criterion(out, img[:, i])
            kld = kl_criterion(mu, logvar, self.batch_size) * self.kl_annealing.get_beta()
            loss += mse + kld
        loss.backward()
        self.optimizer_step()
        return loss
    
    # def histogram_specification(self,img, last_img):
    #     # do histogram specification on img
    #     img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #     last_img = last_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
    #     matched_img = match_histograms(img, last_img, channel_axis=-1)
    #     return torch.tensor(matched_img).permute(2, 0, 1).unsqueeze(0).to(self.args.device)
            
        
    def val_one_step(self, img, label):
        loss = 0
        images = [img[0, 0]]
        with torch.no_grad():
            last = img[:, 0]
            for i in range(1, self.val_vi_len):
                out, mu, logvar = self.forward(last, img[:, i], label[:, i], mode='test')
                images.append(out.squeeze(0))
                
                #last = self.histogram_specification(out.detach(), last)
                last = out.detach()
                mse = self.mse_criterion(out, img[:, i])
                kld = kl_criterion(mu, logvar, self.batch_size)
                loss += mse + kld
                self.writer.add_scalar("PSNR/"+str(self.current_epoch), Generate_PSNR(out, img[:, i]), i)
        return images, loss
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        
        if random.random() < -1:
            transform = transforms.Compose([
                transforms.CenterCrop((400, 800)),
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor()
            ])
            print("crop")
            dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=0.2)
        else:
            transform = transforms.Compose([
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor()
            ])
            dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                    partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        if random.random() < -1:
            transform = transforms.Compose([
                transforms.CenterCrop((256, 512)),
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor()
            ])
            print("crop")
        else:
            transform = transforms.Compose([
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor()
            ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch > self.tfr_sde:
            self.tfr = max(0, self.tfr - self.tfr_d_step)
        
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 200], gamma=0.15)# optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.join(args.save_root, 'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(args.save_root, 'GIF'), exist_ok=True)
    
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=6)
    parser.add_argument('--lr',            type=float,  default=0.0015,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda:1", "cpu"], default="cuda:1")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=500,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=10,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=0.8,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=8,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None, help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=100,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    args = parser.parse_args()
    main(args)
