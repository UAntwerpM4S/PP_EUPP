from typing import Tuple
from math import sqrt, prod
import torch
import torch.nn.functional as F
import einops
import os 
import numpy as np 

class EnsAttention(torch.nn.Module):
    def __init__(self, n_data_shape: Tuple[int, int, int, int], n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        n_channels = n_data_shape[-1]
        
        # Ensure that n_channels is divisible by n_heads
        assert n_channels % n_heads == 0, "n_channels must be divisible by n_heads"
        self.attention_channels = n_channels // n_heads

        self.in_norm_layer = torch.nn.LayerNorm((n_channels,))
        self.qkv_layer = torch.nn.Linear(n_channels, n_channels * 3, bias=False)
        attention_shape = list(n_data_shape[:-1]) + [self.attention_channels]
        attention_dim = prod(attention_shape)
        attention_scale = sqrt(attention_dim)
        
        self.cosine_norm_layer = torch.nn.LayerNorm((attention_dim,))
        self.scaling_factor = torch.nn.Parameter(torch.full((self.n_heads, 1, 1), fill_value=1 / attention_scale))

        self.out_layer = torch.nn.Linear(n_channels, n_channels)
        torch.nn.init.normal_(self.out_layer.weight, std=1E-5)
        
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        b, e, t, h, w, c = in_tensor.shape
        
        normed_tensor = self.in_norm_layer(in_tensor)
        qkv_tensor = self.qkv_layer(normed_tensor)
        qkv_tensor = einops.rearrange(
            qkv_tensor,
            "b e t h w (k n c) -> k b n e (t h w c)",
            k=3, n=self.n_heads, c=self.attention_channels
        )
        normed_qk = self.cosine_norm_layer(qkv_tensor[:2])
        scaled_qk = normed_qk * self.scaling_factor
        attention_tensor = scaled_dot_product_attention_custom(
            scaled_qk[0], scaled_qk[1], qkv_tensor[2], scale=1.
        )
        attention_tensor = einops.rearrange(
            attention_tensor,
            "b n e (t h w c) -> b e t h w (n c)",
            n=self.n_heads, c=self.attention_channels,
            t=t, h=h, w=w
        )
        branch_tensor = self.out_layer(attention_tensor)
        return in_tensor + branch_tensor
        
    
class MLPBlock(torch.nn.Module):
    def __init__(self, n_channels: int, n_mult: int) -> None:
        super().__init__()
        hidden_channels = int(n_channels * n_mult)
        self.norm_layer = torch.nn.LayerNorm((n_channels,))
        self.lin_in = torch.nn.Linear(n_channels, hidden_channels)
        self.activation = torch.nn.GELU()
        self.lin_out = torch.nn.Linear(hidden_channels, n_channels)
        torch.nn.init.normal_(self.lin_out.weight, std=1E-5)
        
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        normed_tensor = self.norm_layer(in_tensor)
        proj_tensor = self.lin_in(normed_tensor)
        activated_tensor = self.activation(proj_tensor)
        branch_tensor = self.lin_out(activated_tensor)
        return in_tensor + branch_tensor
        

class TransformerBlock(torch.nn.Module):
    def __init__(self, n_data_shape: Tuple[int, int, int, int], n_heads: int, mlp_mult: int):  
        super().__init__()
        self.attention_layer = EnsAttention(n_data_shape, n_heads)
        self.mlp_layer = MLPBlock(n_data_shape[-1], mlp_mult)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        attended_tensor = self.attention_layer(in_tensor)
        out_tensor = self.mlp_layer(attended_tensor)
        return out_tensor

class StackedTransformer(torch.nn.Module):
    def __init__(self, num_blocks: int, n_data_shape: Tuple[int, int, int, int], n_heads: int, mlp_mult: int, projection_channels: int):
        super().__init__()
        
        self.projection_layer = torch.nn.Linear(n_data_shape[-1], projection_channels)  # Project 10 -> `projection_channels`
        projected_data_shape = list(n_data_shape[:-1]) + [projection_channels]
        
        # Add multiple projection layers with GELU activation
        
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(n_data_shape=tuple(projected_data_shape), n_heads=n_heads, mlp_mult=mlp_mult)
            for _ in range(num_blocks)
        ])
        
        # Final layer to project back to 1 channel
        self.output_layer = torch.nn.Linear(projection_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection_layer(x)  
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)  
        return x



def scaled_dot_product_attention_custom(query, key, value, scale):
    # #Calculate the dot products between query and key, and scale them
    # query_numpy = query.detach().cpu().numpy()
    # key_numpy = key.detach().cpu().numpy()
    # np.save(os.path.join("./results", "query.npy"), query_numpy)
    # np.save(os.path.join("./results", "key.npy"), key_numpy)
    # Calculate the dot products between query and key, and scale them
    scores = torch.einsum("...qd,...kd->...qk", query, key) / scale
    # Apply softmax to get the attention weights
    attention_weights = F.softmax(scores, dim=-1)
    # Calculate the output by weighting the values
    output = torch.einsum("...qk,...kd->...qd", attention_weights, value)
    return output

def Tformer_prepare(args):
    return StackedTransformer(
        num_blocks=args.num_blocks,  
        n_data_shape=(20, 32, 33, 11),  
        n_heads=args.nheads,  
        mlp_mult=args.mlp_mult,
        projection_channels=args.projection_channels
    )
