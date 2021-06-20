import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional

def elu_feature_map(x):
    return F.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads: int=8,
        eps=1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim = d_model // n_heads
        self.eps = eps

        self.merge = nn.Linear(d_model, d_model)
        self.proj = nn.ModuleList([
            deepcopy(self.merge) for _ in range(3)
            ])
        self.feature_map = elu_feature_map

    def forward(self, queries, keys, values, query_masks, key_masks): # [B, L, D]
        B, L, D = queries.shape
        _, S, _ = keys.shape

        queries, keys, values = [l(x).view(B, -1, self.n_heads, self.dim) for l, x in zip(self.proj, (queries, keys, values))]
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        if query_masks is not None:
            Q = Q * query_masks[:, :, None, None]
        if key_masks is not None:
            K = K * key_masks[:, :, None, None]
            values = values * key_masks[:, :, None, None]
        
        KV = torch.einsum('nshd,nshm->nhmd', K, values)
        Z = 1 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        V = torch.einsum('nlhd,nhmd,nlh->nlhm', Q, KV, Z)

        return self.merge(V.contiguous().view(B, L, -1))

class TransformerLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        dropout=0.1,
        activation='relu'
    ):
        super().__init__()
        
        self.attention = attention
        #FFN
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False), 
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, source, x_masks=None, source_masks=None):
        N, L, E = x.shape
        _, S, _ = source.shape

        msg = self.attention(
            x, 
            source,
            source, 
            query_masks=x_masks, 
            key_masks=source_masks
        )
        msg = self.norm1(msg)

        msg = self.mlp(msg)
        msg = self.norm2(msg)

        return x + msg

class LoFTRModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_names: list,
        dropout: float=0.1,
        activation: str='relu',
    ):
        super().__init__()
        self.d_model = d_model
        
        self.attention = LinearAttention(d_model, n_heads)
        self.layers = nn.ModuleList([
            TransformerLayer(self.attention,
                            d_model,
                            dropout,
                            activation)
            for _ in range(len(layer_names))
        ])

        self.names = layer_names
        self._rest_parameters()
    def _rest_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        desc0, desc1,
        mask0=None, mask1=None,
    ):
        """
        desc0 desc1 ---> [N, L, D] [N, S, D]
        mask0 mask1 ---> [N, L] [N, S]
        """
        for name, layer in zip(self.names, self.layers):
            if name == 'self':
                desc0 = layer(desc0, desc0, mask0, mask0)
                desc1 = layer(desc1, desc1, mask1, mask1)
            else:
                desc0 = layer(desc0, desc1, mask0, mask1)
                desc1 = layer(desc1, desc0, mask1, mask0)

        return desc0, desc1