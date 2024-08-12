import torch
import torch.nn as nn

class BiLSTMLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, num_lstm_layer=3, num_classes=3, hidden_size=356):
        super(BiLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(
            input_size=d_model, bidirectional=True, num_layers=num_lstm_layer, 
            hidden_size=hidden_size, batch_first=True
        )

        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, _ = self.bilstm(x)
        hidden_layer = self.dropout(output)
        logit = self.fc(hidden_layer)
        prob = nn.functional.softmax(logit, dim=-1)

        return prob
