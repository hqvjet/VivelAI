import torch
import torch.nn as nn
import json

with open('models/GRU/config.json', 'r') as file:
    config = json.load(file)

phobert_config = config['phobert']

class GRU(nn.Module):
    def __init__(self, device, input_shape, dropout=0.1, num_classes=3):
        super(GRU, self).__init__()
        config = phobert_config
        self.model_name = 'GRU'
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        self.gru = nn.GRU(input_size=input_shape[-1], hidden_size=self.hidden_size,\
                            num_layers=self.num_layers, device=device, dropout=dropout,\
                            batch_first=True)

        first_emb = 128
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        self.fc2 = nn.Linear(first_emb, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.unsqueeze(1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc1(out)
        # out = self.fc2(out)
        
        out = self.softmax(out)
        
        return out
