import torch.nn as nn
import torch
import math

#TODO1       
               
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV linear layers, will be split into num_heads later
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        
        # Output linear layer
        self.out_linear = nn.Linear(dim, dim)
        
        # Dropout layer
        self.attn_drop = nn.Dropout(attn_drop)
        

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_image_tokens, _ = x.shape
        
        # get q, k, v matrices
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # reshape to split by heads and then transpose to get the shape (batch_size, num_heads, num_image_tokens, head_dim)
        q = q.view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # do matrix multiplication for each head
        # q dimension: (batch_size, num_heads, num_image_tokens, head_dim)
        # k dimension: (batch_size, num_heads, head_dim, num_image_tokens)
        # scores dimension: (batch_size, num_heads, num_image_tokens, num_image_tokens)
        # stabilize the gradients by scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # attn_weights dimension: (batch_size, num_heads, num_image_tokens, num_image_tokens)
        # v dimension: (batch_size, num_heads, num_image_tokens, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # concatenate heads and apply the final linear layer
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_image_tokens, self.dim)
        output = self.out_linear(attn_output)
        
        return output

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    