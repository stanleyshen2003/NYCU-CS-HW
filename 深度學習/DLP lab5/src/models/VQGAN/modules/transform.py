import torch.nn as nn
import torch
import functools
from .layers import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish, UpSampleBlock

class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        channels = configs['enc_channels']
        num_res_blocks = 2
        resolution = configs['img_resolution']
        attn_resolutions = [configs['latent_resolution']]

        layers = [nn.Conv2d(configs['image_channels'], channels[0], 3, 1, 1)]

        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            #print("enc i:",i,"block_in",in_channels,"block_out",out_channels)
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i ==2 or i==4:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        
        layers.extend([
            ResidualBlock(channels[-1], channels[-1]),
            NonLocalBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1]),
            GroupNorm(channels[-1]),
            Swish(),
            nn.Conv2d(channels[-1], configs['latent_dim'], 3, 1, 1)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        attn_resolutions = [configs['latent_resolution']]
        channels = configs['dec_channels']
        num_ch = len(channels)
        curr_res=configs['latent_resolution']

        layers = [nn.Conv2d(configs['latent_dim'], channels[-1], kernel_size=3, stride=1, padding=1),
                  ResidualBlock(channels[-1], channels[-1]),
                  NonLocalBlock(channels[-1]),
                  ResidualBlock(channels[-1], channels[-1])
                  ]
        in_channels = channels[num_ch-1]
        for i in reversed(range(num_ch)):
            out_channels = channels[i]
            #print("dec i:",i,"block_in",in_channels,"block_out",out_channels)
            for j in range(3):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if curr_res in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i == 4 or i == 2:
                layers.append(UpSampleBlock(in_channels))
                curr_res = curr_res * 2

        layers.append(GroupNorm(in_channels))
        layers.append(nn.Conv2d(in_channels, configs['image_channels'], kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class Codebook(nn.Module):
    """
    Codebook mapping: takes in an encoded image and maps each vector onto its closest codebook vector.
    Metric: mean squared error = (z_e - z_q)**2 = (z_e**2) - (2*z_e*z_q) + (z_q**2)
    """

    def __init__(self, configs):
        super().__init__()
        self.num_codebook_vectors = configs['num_codebook_vectors']
        self.latent_dim = configs['latent_dim']
        self.beta = configs['beta']

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)
        #1024*256
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        #b h w 256  #b*h*w,256

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
        #     torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
                
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1) #b*256
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        #b,h,w,256
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()  # moving average instead of hard codebook remapping

        z_q = z_q.permute(0, 3, 1, 2)
        #b,c,h,w
        return z_q, min_encoding_indices, loss


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (https://arxiv.org/pdf/1611.07004.pdf)
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, configs, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            

        num_filters_last = configs['num_filters_last']
        n_layers = configs['n_layers']

        kernel_size = 4
        padding_size = 1
        sequence = [nn.Conv2d(configs['image_channels'], num_filters_last, kernel_size, stride=2, padding=padding_size),
                    nn.LeakyReLU(0.2)]
        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            sequence += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, kernel_size,
                          2 if i < n_layers else 1, padding_size, bias=use_bias),
                norm_layer(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size, 1, padding_size)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)