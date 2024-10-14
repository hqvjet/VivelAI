import torch
import torch.nn as nn
import json

with open('models/LSTM/config.json', 'r') as file:
    config = json.load(file)

config = config['phobert']
num_classes = 3

class BiLSTM(nn.Module):
    def __init__(self, device, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.model_name = 'BiLSTM'
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,\
                            num_layers=self.num_layers, device=device, dropout=dropout,\
                            batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(self.hidden_size*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Init hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        x = x.unsqueeze(1)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last hidden state
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        # Apply softmax for classification
        out = self.softmax(out)
        
        return out
