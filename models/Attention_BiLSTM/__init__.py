import torch
import torch.nn as nn
import torch.nn.functional as F
import json

with open('models/Attention_BiLSTM/config.json', 'r') as file:
    config = json.load(file)

phobert_config = config['phobert']
phow2v_config = config['phow2v']
num_classes = 3

class AttentionBiLSTM(nn.Module):
    def __init__(self, device, input_shape, emb_tech, dropout=0.1):
        super(AttentionBiLSTM, self).__init__()
        config = phobert_config if emb_tech == 1 else phow2v_config
        self.model_name = 'AttentionBiLSTM'
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.emb_tech = emb_tech

        self.lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=self.hidden_size,\
                            num_layers=self.num_layers, device=device, dropout=dropout,\
                            batch_first=True, bidirectional=True)

        first_emb = 256 if emb_tech == 1 else 32
        self.attention_w = nn.Linear(self.hidden_size*2, 1)
        self.fc1 = nn.Linear(self.hidden_size*2, 256)
        self.fc2 = nn.Linear(first_emb, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def attention_layer(self, lstm_out):
        """
        lstm_out: [batch_size, 1, hidden_dim*2]
        Returns: weighted sum of lstm outputs
        """
        # Compute attention scores
        attn_scores = self.attention_w(lstm_out)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1)              # Normalize scores: [batch_size, seq_len]
        weighted_out = attn_weights * lstm_out
        
        return weighted_out

    def forward(self, x):
        if self.emb_tech == 1:
            x = x.unsqueeze(1)

            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.attention_layer(out)

        else:
            _, (hn, _) = self.lstm(x)
            out = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        # Apply softmax for classification
        out = self.softmax(out)
        
        return out
