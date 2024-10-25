import torch
import torch.nn as nn

num_classes = 3

class FC(nn.Module):
    def __init__(self, input_shape, device, emb_tech, dropout=0.1):
        super(FC, self).__init__()
        self.model_name = 'FullyConnected'
        self.fc1 = nn.Linear(input_shape[-1], 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.fc4 = nn.Linear(input_shape[-1], num_classes)
        self.sm = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_tech = emb_tech

        self.pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        if self.emb_tech == 2:
            x = self.pool(x.permute(0, 2, 1)).squeeze(2)
        out = self.dropout(x)
        if self.emb_tech == 1:
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
        else:
            out = self.fc4(out)
        return self.sm(out)
