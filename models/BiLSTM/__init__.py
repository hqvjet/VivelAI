import torch
import torch.nn as nn
import json

with open('models/BiLSTM/config.json', 'r') as file:
    config = json.load(file)

phobert_config = config['phobert']

class BiLSTM(nn.Module):
    def __init__(self, device, input_shape, dropout=0.1, num_classes=3):
        super(BiLSTM, self).__init__()
        config = phobert_config
        self.model_name = 'BiLSTM'
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        self.lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=self.hidden_size,\
                            num_layers=self.num_layers, device=device, dropout=dropout,\
                            batch_first=True, bidirectional=True)

        first_emb = 128 if emb_tech == 1 else 32
        self.fc1 = nn.Linear(self.hidden_size*2, num_classes)
        self.fc2 = nn.Linear(first_emb, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.unsqueeze(1)

        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc1(out)
        # out = self.fc2(out)
        
        out = self.softmax(out)
        
        return out
