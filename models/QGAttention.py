import torch
from torch import nn
from torchtune.modules.attention import CausalSelfAttention
from torchtune.modules import RotaryPositionalEmbeddings
from grouped_query_attention_pytorch.attention import MultiheadGQA


class PositionalEmbeddings(nn.Module):
    def forward(self, x, input_pos=None):
        # Normally, you'd apply positional embeddings here, but we're keeping it simple
        return x



class Attention_QGA(nn.Module):
    def __init__(self, embed_dim, num_heads=12):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.pos_embeddings = RotaryPositionalEmbeddings(dim=head_dim)
        self.embed_dim = embed_dim
        q_proj = nn.Linear(embed_dim, embed_dim)
        k_proj = nn.Linear(embed_dim, embed_dim)
        v_proj = nn.Linear(embed_dim, embed_dim)
        output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.attention_layer = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,  # Use the same number of heads for key/value
            head_dim=head_dim,  # Correctly use head_dim here
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=output_proj,
            pos_embeddings=self.pos_embeddings,
            attn_dropout=0.1
        )
    
    def forward(self, x):
        x_shape = x.shape
        output = self.attention_layer(x.view(x.shape[0] * x.shape[1], -1, self.embed_dim))
        output = self.norm(output)
        return output.view(x_shape)


class MultiheadGQAConv(nn.Module):
    def __init__(self, embedding_dim,channels):
        super().__init__()
        
        self.attention_layer = MultiheadGQA(embed_dim=embedding_dim, query_heads=4, kv_heads=2, device="cuda")
        self.convK = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=1,bias=False)
        self.convQ = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=2,stride=2,bias=False)
        self.convV = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=2,stride=2,bias=False)
        self.embedding_dim = embedding_dim

        


    def forward(self, x):
        k = self.convK(x)
        q = self.convQ(x)
        v = self.convV(x)
        




        x_shape = x.shape

        k = k.contiguous().view(x_shape[0],-1,self.embedding_dim) 
        q = q.contiguous().view(x_shape[0],-1,self.embedding_dim) 
        v = v.contiguous().view(x_shape[0],-1,self.embedding_dim) 
        output = self.attention_layer(k,q,v)[0]

        #output = self.norm(output)
        return output.view(x_shape)