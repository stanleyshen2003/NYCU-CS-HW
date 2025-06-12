import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import random


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = 0.3#configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])
        

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        zq, codebook_indices, q_loss = self.vqgan.encode(x)
        return zq, codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda ratio: 1 - ratio
        elif mode == "cosine":
            return lambda ratio: math.cos(math.pi / 2 * ratio)
        elif mode == "square":
            return lambda ratio: 1 - ratio ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        '''
        mask ratio can be adjusted
        '''
        # z_indices=None #ground truth
        _, z_indeces = self.encode_to_z(x)
        
        # reshape z_indeces to (batch_size, num_image_tokens)
        z_indeces = z_indeces.view(-1, self.num_image_tokens)
        
        #print(z_indeces.shape)
        ratio = torch.rand(1)
        ratio = (ratio * 0.5).item()
        mask = torch.rand_like(z_indeces, dtype=float) < ratio
        # assign value 1024 to the masked token
        masked_z = torch.where(mask, 1024, z_indeces)
        logits = self.transformer(masked_z)
        return logits, z_indeces
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indeces, mask, ratio, mask_num):
        masked_z = torch.where(mask, 1024, z_indeces)
        
        logits = self.transformer(masked_z)
        # get softmax probability
        logits = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
        
        # find max probability tokens
        z_indeces_predict_prob, z_indeces_predict = torch.max(logits, dim=-1)

        # add gumbel noise to the confidence
        g =  -torch.empty_like(z_indeces_predict_prob).exponential_().log()
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indeces_predict_prob  + temperature * g
        
        # find the threshold of confidence
        new_confidence = torch.where(mask, confidence, float('inf'))
        z_sort = torch.argsort(new_confidence, descending=False)
        n = ratio * mask_num
        confidence_threshold = new_confidence[0, z_sort[0,int(n)]]
        
        # put the original token values back if its on the mask
        z_indeces_predict = torch.where(mask, z_indeces_predict, z_indeces)
        mask_bc = new_confidence < confidence_threshold
        return z_indeces_predict.squeeze(0), mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
