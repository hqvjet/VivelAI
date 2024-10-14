import torch
import torch.nn as nn

num_classes = 3

class FC(nn.Module):
    def __init__(self, emb_dim, device, dropout=0.1):
        super(FC, self).__init__()
        self.model_name = 'FullyConnected'
        self.fc1 = nn.Linear(emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.sm = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.fc1(x)
        drop1 = self.dropout(out)
        out = self.fc2(drop1)
        out = self.fc3(out)
        return self.sm(out)
