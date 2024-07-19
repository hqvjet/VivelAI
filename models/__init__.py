from models.bert import trainBERT
import torch
import numpy as np

def startTraining(device):
    key = input('Choose feature source:\n1. PhoBERT Detail\n2. PhoBERT Total\n3. PhoW2V\n')

    print('Loading extracted feature')

    if key == '1':
        source = 'phobert_detail'
    elif key == '2':
        source = 'phobert_total'
    elif key == '3':
        source = 'phow2v'
    else:
        print('Wrong source, please try again')

    title = np.load(f'res/features/{source}_title_features.npy')
    content = np.load(f'res/features/{source}_content_features.npy')
    
    title = torch.from_numpy(title)
    content = torch.from_numpy(content)

    key = input('Choose one of these model to train:\n1. BERT\nYour Input: ')

    if key == '1':
        trainBERT(device, title, content)
    else:
        print('Wrong key of model, please choose again')

