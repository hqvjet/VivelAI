import torch
import torch.nn as nn
from models.CNN import CNN2d
from models.LSTM import LSTM

num_classes = 3

class CNNnLSTM(nn.Module):
    def __init__(self, device, input_shape, emb_tech, useTitle, dropout=0.0):
        super(CNNnLSTM, self).__init__()
        self.model_name = 'Ensemble_CNN_LSTM'
        self.cnn = CNN2d(device, input_shape, emb_tech, dropout)
        self.lstm = LSTM(device, input_shape, emb_tech, dropout)
        self.emb_tech = emb_tech
        model_direction = 'phobert' if emb_tech == 1 else 'phow2v'
        direction = 'with_title' if useTitle else 'no_title'
        self.cnn.load_state_dict(torch.load(f'res/models/{direction}/{model_direction}/CNN.pth'))
        self.lstm.load_state_dict(torch.load(f'res/models/{direction}/{model_direction}/LSTM.pth'))

        self.fc1 = nn.Linear(num_classes, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(x)

        cnn_fc = self.fc1(cnn_out)
        lstm_fc = self.fc1(lstm_out)

        out = (cnn_fc + lstm_fc) / 2
        out = self.fc2(out)
        out = self.fc3(out)

        return self.softmax(out)
