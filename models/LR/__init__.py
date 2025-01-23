from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib
from constant import DRIVE_PATH

class LR(nn.Module):
    def __init__(self, extract_model, dataset):
        super(LR, self).__init__()
        self.model_name = 'Logistic_Regression'
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self.extract_model = extract_model
        self.dataset = dataset

    def forward(self, x, y=None, train=True):
        if train:
            self.model.fit(x, y)
            joblib.dump(self.model, (f'res/models/{self.extract_model}/{self.dataset}/{self.model_name}.pkl'))
            joblib.dump(self.model, (f'{DRIVE_PATH}/models/{self.extract_model}/{self.dataset}/{self.model_name}.pkl'))

        else:
            self.model = joblib.load(f'res/models/{self.extract_model}/{self.dataset}/{self.model_name}.pkl')

            predicted = self.model.predict(x)
            return predicted
