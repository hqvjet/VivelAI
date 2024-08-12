import torch.nn as nn
import torch

class AddNormalizeLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(AddNormalizeLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

        self.layer_norm = self.layer_norm

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)
