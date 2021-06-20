import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbedding2D(nn.Module):
    def __init__(
        self, 
        d_model: int,
        max_size: tuple=(255, 255),
        temperature: float=10000.,
        
    ):
        super().__init__()
        # kpts1:[x, y], kpts2:[x, y]
        dim = d_model // 2
        pe = torch.zeros((d_model, *max_size)).float() #[D, H, W]
        pe.requires_grad = False

        x_pos = torch.ones(*max_size).cumsum(dim=1).float().unsqueeze(0)
        y_pos = torch.ones(*max_size).cumsum(dim=0).float().unsqueeze(0)
    
        div_term = torch.exp(
                torch.arange(0, dim, 2).float() * (-math.log(temperature) / dim)
                )
        div_term = div_term[:, None, None]

        pe[0::4, :, :] = torch.sin(x_pos * div_term)
        pe[1::4, :, :] = torch.cos(x_pos * div_term)
        pe[2::4, :, :] = torch.sin(y_pos * div_term)
        pe[3::4, :, :] = torch.cos(y_pos * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor):
        return self.pe[:, :, :x.size(2), :x.size(3)]

class PositionEmbeding1D(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int=64,
        temperature: float=10000.,
    ):
        super().__init__()
        dim = d_model
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(1, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(temperature) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return self.pe[:, :x.size(1)]
