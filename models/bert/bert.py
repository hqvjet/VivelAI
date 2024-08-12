import torch.nn as nn
import torch
import math

from models.bert.positional_encoding import PositionalEncodingLayer
from models.bert.mha import MultiHeadAttentionLayer
from models.bert.add_norm import AddNormalizeLayer
from models.bert.feedforward import FeedForwardLayer
from models.classifier.BiLSTM import BiLSTMLayer

class BERT(nn.Module):
    def __init__(self, d_model, max_len, device, num_head=15, dropout=0.1, debug=False, num_layer=24):
        super(BERT, self).__init__()

        self.dropout = nn.Dropout(dropout).to(device)
        self.num_head = num_head
        self.device = device
        self.d_model = d_model
        self.max_len = max_len
        self.debug = debug
        self.cls = nn.Parameter(torch.randn(1, self.d_model)).to(device)
        self.PositionalEncodingLayer = PositionalEncodingLayer(max_len=self.max_len+1, d_model=self.d_model).to(device)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionLayer(head_num=self.num_head, d_model=self.d_model).to(device),
                AddNormalizeLayer(d_model=self.d_model).to(device),
                FeedForwardLayer(d_ff=4*self.d_model, d_model=self.d_model).to(device),
                AddNormalizeLayer(d_model=self.d_model).to(device),
            ])
            for _ in range(num_layer)
        ])
        self.classifier = BiLSTMLayer(d_model=self.d_model, dropout=dropout, num_lstm_layer=2, num_classes=3, hidden_size=356).to(device)

    def appendCLS(self, input, size):
        # Add CLS to start of sentence
        cls_toks = self.cls.expand(size, 1, -1)
        return torch.cat((cls_toks, input), dim=1)

    def forward(self, x):
        x = self.appendCLS(x, x.size(0))

        drop_2 = self.PositionalEncodingLayer(x)

        for mha, add_norm1, ff, add_norm2 in self.layers:
           # Multi-head attention: PE
            x_mha = mha(drop_2)

            # Add & Norm: Concat(PE, MHA)
            x_an1 = add_norm1(drop_2, x_mha)
            drop_1 = self.dropout(x_an1)
            # # FeedForward: Add & Norm
            x_ff = ff(drop_1)
            # # Add & Norm: Concat(Previous Add & Norm, FeedForward)
            x_an2 = add_norm2(x_ff, drop_1)
            drop_2 = self.dropout(x_an2)

            del x_mha
            del x_ff
            del x_an1
            del x_an2
            del drop_1

            torch.cuda.empty_cache()

        return self.classifier(drop_2[:, 0, :])
