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
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        self.fc = nn.Linear(self.hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (batch, emb)

        out = []
        for sentence_cls in x:
            total_block = []
            print(sentence_cls.size())
            for filter in filters:
                conv1d_layer = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)(sentence_cls)
                relu = F.relu(conv1d_layer)
                maxpool_layer = nn.MaxPool2d(kernel_size=(2), stride=2)(conv2d_layer)
                total_block.append(maxpool_layer)
            total_block = torch.tensor(total_block)
            print(total_block.size())
            drop = self.dropout(total_block)

            out.append(self.softmax(drop))

        return torch.tensor(out)
