from models.bert import trainBERT
import torch
import numpy as np
import pandas as pd

def startTraining(device):
    key = input('Choose feature source:\n1. PhoBERT Detail\n2. PhoBERT Total\n3. PhoW2V\nYour Input: ')

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
    # data = pd.read_excel('res/datasets.xlsx', engine='openpyxl')
    data = pd.read_csv('res/data.csv')
    rating = data['Rating'].apply(int)
    rating -= 1
    rating = torch.tensor(rating)[:8804]
    o, t, tr = 0, 0, 0
    for r in rating:
        if r == 0:
            o += 1
        elif r == 1:
            t += 1
        else:
            tr += 1

    print(f'Negative: {o}, Neural: {t}, Positive: {tr}')

    rating = torch.nn.functional.one_hot(rating, num_classes=3)
    
    title = torch.from_numpy(title)
    content = torch.from_numpy(content)

    print('Loading Done')
    print(f'Title Shape: {title.size()}')
    print(f'Content Shape: {content.size()}')
    print(f'Label Shape: {rating.size()}')

    key = input('Choose one of these model to train:\n1. BERT\nYour Input: ')

    if key == '1':
        trainBERT(device, title, content, rating)
    else:
        print('Wrong key of model, please choose again')
