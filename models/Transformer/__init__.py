import torch
import torch.nn as nn
import json

with open('models/Transformer/config.json', 'r') as file:
    config = json.load(file)

phobert_config = config['phobert']
phow2v_config = config['phow2v']
num_classes = 3

class Transformer(nn.Module):
    def __init__(self, device, input_shape, emb_tech, dropout=0.1):
        super(Transformer, self).__init__()
        config = phobert_config if emb_tech == 1 else phow2v_config
        self.model_name = 'Transformer'
        self.hidden_size = input_shape[-1]
        self.num_layers = config['num_layers']
        self.emb_tech = emb_tech

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_shape[-1], nhead=8,
                dim_feedforward=input_shape[-1] * 4,
                dropout=dropout,
                activation='relu'
            ),
            num_layers=self.num_layers
        )

        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.emb_tech == 1:
            x = x.unsqueeze(1)

        trans_out = self.encoder(x.permute(1, 0, 2))

        cls_tok = trans_out[0, :, :]
        out = self.fc(cls_tok)
        out = self.softmax(out)

        return out
