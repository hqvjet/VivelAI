import torch
import torch.nn as nn
import json

with open('models/LSTM/config.json', 'r') as file:
    config = json.load(file)

phobert_config = config['phobert']
phow2v_config = config['phow2v']
num_classes = 3

class LSTM(nn.Module):
    def __init__(self, device, input_shape, emb_tech, dropout=0.1):
        super(LSTM, self).__init__()
        config = phobert_config if emb_tech == 1 else phow2v_config
        self.model_name = 'LSTM'
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.emb_tech = emb_tech

        self.lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=self.hidden_size,\
                            num_layers=self.num_layers, device=device, dropout=dropout,\
                            batch_first=True)

        first_emb = 128 if emb_tech == 1 else 32
        self.fc1 = nn.Linear(self.hidden_size, num_classes)
        self.fc2 = nn.Linear(first_emb, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.emb_tech == 1:
            x = x.unsqueeze(1)

            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
        
        else:
            _, (hn, _) = self.lstm(x)
            out = hn[-1]
        
        out = self.dropout(out)
        out = self.fc1(out)
        # out = self.fc2(out)
        
        # Apply softmax for classification
        out = self.softmax(out)
        
        return out
