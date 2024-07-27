import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, device, head_num=16, d_model=300):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % head_num == 0, "d_model must be divisible by head_num"

        self.d_model = d_model
        self.head_num = head_num
        self.d_head = d_model // head_num
        self.device = device

        # Create random weights matrices
        self.W_q = nn.ModuleList([nn.Linear(self.d_model, self.d_head) for _ in range(self.head_num)])
        self.W_k = nn.ModuleList([nn.Linear(self.d_model, self.d_head) for _ in range(self.head_num)])
        self.W_v = nn.ModuleList([nn.Linear(self.d_model, self.d_head) for _ in range(self.head_num)])

        self.W_o = nn.Linear(self.d_model, self.d_model)
        
        self.W_q = self.W_q.to(self.device)
        self.W_k = self.W_k.to(self.device)
        self.W_v = self.W_v.to(self.device)
        self.W_o = self.W_o.to(self.device)

    def calculate_Score(self, q, k, mask=None):
        # Use transpose at k because of Matrix Multiplying
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores /= math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        return nn.functional.softmax(scores, dim=-1)

    def multiplying_value(self, scores, v):
        return torch.matmul(scores, v)

    def forward(self, x, mask=None):

        z_out = []

        for i in range(self.head_num):
            q = self.W_q[i](x)
            k = self.W_k[i](x)
            v = self.W_v[i](x)

            z = self.calculate_Score(q, k)
            z = self.multiplying_value(z, v)

            z_out.append(z)

        multihead_out = torch.cat(z_out, dim=-1)

        return self.W_o(multihead_out)
