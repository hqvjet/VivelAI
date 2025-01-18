from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib
from constant import DRIVE_PATH

class SVM(nn.Module):
    def __init__(self, emb_tech, useTitle):
        super(SVM, self).__init__()
        self.model_name = 'SVM'
        self.model = SVC(C=0.1, kernel='rbf', gamma='auto')
        self.emb_tech = emb_tech
        self.direction = 'with_title' if useTitle else 'no_title'
        self.model_direction = 'phobert' if emb_tech == 1 else 'phow2v'

    def forward(self, x, y=None, train=True):
        if self.emb_tech == 2:
            x = x.reshape(x.shape[0], -1)

        if train:
            self.model.fit(x, y)

            joblib.dump(self.model, (f'res/models/{self.direction}/{self.model_direction}/{self.model_name}_viso.pkl'))
            joblib.dump(self.model, (f'{DRIVE_PATH}/models/{self.direction}/{self.model_direction}/{self.model_name}_viso.pkl'))

        else:
            self.model = joblib.load(f'res/models/{self.direction}/{self.model_direction}/{self.model_name}_viso.pkl')

            predicted = self.model.predict(x)
            return predicted
