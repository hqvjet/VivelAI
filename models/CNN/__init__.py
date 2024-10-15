import torch
import torch.nn as nn
import torch.nn.functional as F
import json

with open('models/CNN/config.json', 'r') as file:
    config = json.load(file)

config = config['phobert']
num_classes = 3
filters = [2, 3, 4]

class CNN2d(nn.Module):
    def __init__(self, device, dropout=0.0):
        super(CNN2d, self).__init__()
        self.model_name = 'CNN'
        self.num_filter = config['num_filter']
        input_dim = config['input_size']

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=self.num_filter, kernel_size=fs) 
            for fs in filters
        ])

        self.fc1 = nn.Linear(self.num_filter * len(filters), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (batch, emb)
        x = x.unsqueeze(1) # (batch, 1, emb)

        conved = [F.relu(conv(x)).squeeze(2) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        drop = self.dropout(cat)
        fc1 = self.fc1(drop)
        fc2 = self.fc2(fc1)

        return self.softmax(fc2)
