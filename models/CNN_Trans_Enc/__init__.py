import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_classes = 2

class CNN_Trans_Enc(nn.Module):
    def __init__(self, input_shape, emb_tech, kerner_size=[2,3,4,5], cnn_filter=256, trans_layer=2, num_head=8, dropout=0.1):
        super(CNN_Trans_Enc, self).__init__()
        self.model_name = 'CNN_Trans_Enc'
        self.emb_tech = emb_tech

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_shape[-1], out_channels=128, kernel_size=k, padding='same', padding_mode='zeros')
            for k in kerner_size
        ])

        enc_layer = nn.TransformerEncoderLayer(d_model=128*len(kerner_size), nhead=num_head, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=trans_layer)

        self.fc = nn.Linear(len(kerner_size) * cnn_filter, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.size())
        x = x.unsqueeze(2)
        print(x.size())
        cnn_out = [F.relu(conv(x)).squeeze(2) for conv in self.convs]
        for i in cnn_out:
            print(i.size())
        cnn_out = torch.cat(cnn_out, dim=1)

        cnn_out = cnn_out.unsqueeze(1)
        out = self.trans(cnn_out).squeeze(1)
        print(out.size())

        out = self.fc(out)
        return self.softmax(out)
