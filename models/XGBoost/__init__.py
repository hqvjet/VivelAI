import xgboost as xgb
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from constant import DRIVE_PATH

with open('models/XGBoost/config.json', 'r') as file:
    config = json.load(file)

config = config['phobert']

class XGBoost(nn.Module):
    def __init__(self, extract_model):
        super(XGBoost, self).__init__()
        self.model_name = 'XGBoost'
        self.params = {
            'objective': config['objective'],
            'num_class': config['num_class'],
            'max_depth': config['max_depth'],
            'eta': config['eta'],
            'eval_metric': config['eval_metric'],
            'tree_method': config['tree_method'],
            'device': 'cuda'
        }
        self.extract_model = extract_model
        self.num_boost_round = config['num_boost_round']
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, y=None, train=True):
        if train:
            data = xgb.DMatrix(x, label=y)

            model = xgb.train(self.params, data, num_boost_round=self.num_boost_round)
            print('XGB training has done!')

            # Save model
            model.save_model(f'res/models/{self.extract_model}/{self.model_name}.json')
            model.save_model(f'{DRIVE_PATH}/models/{self.extract_model}/{self.model_name}.json')

        else:
            model = xgb.Booster()
            model.load_model(f'res/models/{self.extract_model}/{self.model_name}.json')

            data = xgb.DMatrix(x)   
            predicted = model.predict(data)
            return predicted
