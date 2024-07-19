import torch.nn as nn
import torch
import math

class PositionalEncodingLayer(nn.Module):
    def __init__(self, device, max_len=200, d_model=300):
        super(PositionalEncodingLayer, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        self.device = device
        
    def forward(self, x):
        """
        x: Tensor, shape [record, max_len, d_model]
        """
        self.pe = self.pe.to(self.device)
        x += self.pe
        self.pe = self.pe.cpu()
        return x 
