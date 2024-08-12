import torch.nn as nn
import torch

class FeedForwardLayer(nn.Module):
    def __init__(self, d_ff, d_model):
        super(FeedForwardLayer, self).__init__()

        self.extend = nn.Linear(d_model, d_ff)
        self.narrow = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.narrow(nn.functional.relu(self.extend(x)))
