import torch
import torch.nn as nn
from .modules.transform import Encoder, Decoder, Codebook

__all__ = [
    "VQGAN"
]

class VQGAN(nn.Module):
    def __init__(self, configs):
        super(VQGAN, self).__init__()
        
        dim = configs['latent_dim']
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)
        self.codebook = Codebook(configs)
        self.quant_conv = nn.Conv2d(dim, dim, 1)
        self.post_quant_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        #b,c,h,w   
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        #b,c,h,w    b*c
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images

    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        #discriminator weight=0.8
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path), strict=True)
        print("Loaded Checkpoint for VQGAN....")

