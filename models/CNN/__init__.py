import torch
import torch.nn as nn
import torch.nn.functional as F
import json

with open('models/CNN/config.json', 'r') as file:
    config = json.load(file)

config = config['phobert']
num_classes = 2
filters = [2, 3, 4]

class CNN2d(nn.Module):
    def __init__(self, device, input_shape, emb_tech, dropout=0.0):
        super(CNN2d, self).__init__()
        self.model_name = 'CNN'
        self.emb_tech = emb_tech
        self.num_filter = config['num_filter']
        self.input_shape = input_shape

        if emb_tech == 2:
            self.convs = nn.ModuleList([
                nn.Conv2d(in_channels=1, out_channels=self.num_filter, kernel_size=(fs, self.input_shape[2]))
                for fs in filters
            ])
            self.fc1 = nn.Linear(self.num_filter * len(filters), 128)
            self.fc2 = nn.Linear(128, num_classes)

        else:
            self.convs = nn.Conv2d(in_channels=1, out_channels=self.num_filter, kernel_size=(1, self.input_shape[1]))
            self.fc1 = nn.Linear(self.num_filter, 32)
            self.fc2 = nn.Linear(32, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.emb_tech == 1:
            x = x.unsqueeze(1).unsqueeze(3)
            x = x.permute(0, 1, 3, 2)

            conv_out = F.relu(self.convs(x))
            x = F.max_pool2d(conv_out, kernel_size=(conv_out.shape[2], 1))

        else:
            x = x.unsqueeze(3)
            x = x.permute(0, 3, 1, 2)

            conv_blocks = []
            for conv in self.convs:
                conv_out = F.relu(conv(x))
                pool_out = F.max_pool2d(conv_out, 
                                        kernel_size=(conv_out.shape[2], 1))
                conv_blocks.append(pool_out)
            
            x = torch.cat(conv_blocks, dim=1)

        x = x.view(x.size(0), -1)
        
        drop = self.dropout(x)
        out = self.fc1(drop)
        out = self.fc2(out)

        return self.softmax(out)
