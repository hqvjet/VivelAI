import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.LSTM import LSTM

with open('models/global_config.json', 'r') as file:
    config = json.load(file)

batch_size = config.get('batch_size')
num_epoch = config.get('num_epoch')

def temp_func(title, content, rating):
    threshold = 381
    neg, neu, pos = 0, 0, 0
    temp_title, temp_content, temp_rating = [], [], []

    def add(index):
        temp_title.append(title[index])
        temp_content.append(content[index])
        temp_rating.append(rating[index])

    i = 0
    while neg < 381 or neu < 381 or pos < 381:
        if rating[i] == 0 and neg < 381:
            neg += 1
            add(i)
        elif rating[i] == 1 and neu < 381:
            neu += 1
            add(i)
        elif rating[i] == 2 and pos < 381:
            pos += 1
            add(i)

        i += 1

    return np.array(temp_title), np.array(temp_content), np.array(temp_rating)

def convert_start_to_sentiment(data):
    res = []
    for i in data:
        if i < 3:
            res.append(0)
        elif i > 3:
            res.append(2)
        else:
            res.append(1)

    return res

def startTraining(device):
    key = input('Choose feature source:\n1. PhoBERT\n2. PhoW2V\nYour Input: ')

    print('Loading extracted feature')

    if key == '1':
        source = 'phobert'
    elif key == '2':
        source = 'phow2v'
    else:
        print('Wrong source, please try again')

    title = np.load(f'res/features/{source}_title_features.npy')
    content = np.load(f'res/features/{source}_content_features.npy')
    data = pd.read_csv('res/data_service.csv')
    rating = data['rating'].apply(int)
    rating = convert_start_to_sentiment(rating)
    title, content, rating = temp_func(title, content, rating)
    rating = torch.from_numpy(rating)

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

    key = input('Choose one of these classification to train:\n1. LSTM\n2.BiLSTM\n3. CNN\n4. XGBoost\nYour Input: ')

    data = torch.cat((title, content), dim=-1)

    if key == '1':
        train(LSTM(device=device, dropout=0.0), input=data, output=rating, device=device)      
    else:
        print('Wrong key of model, please choose again')

def train(model, input, output, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(), lr=0.00001)

    # Splitting dataset
    train_size = int(0.8*input.size(0))
    val_size = int(0.1*train_size)
    train_size -= val_size
    test_size = int(0.2*input.size(0))

    # dataset = TensorDataset(input, output)

    train_data = DataLoader(TensorDataset(input[:train_size], output[:train_size]), batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(TensorDataset(input[train_size:train_size+val_size], output[train_size:train_size+val_size]), batch_size=batch_size, shuffle=True)
    test_data = DataLoader(TensorDataset(input[-1*test_size:], output[-1*test_size:]), batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoch):
        train_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epoch}:")
        val_bar = tqdm(valid_data, desc=f"Epoch {epoch + 1}/{num_epoch}:")
        # test_bar = tqdm(test_data, desc=f"Epoch {epoch + 1}/{num_epoch}:")

        model.train()
        for batch in train_bar:
            input_ids = batch[0].to(device)
            true_output = batch[1].float().to(device)
        
            pred_output = model(input_ids)
            loss = criterion(pred_output, true_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        total_val_loss = 0
        total = 0
        correct = 0

        print(())
        for batch in val_bar:
            input_ids = batch[0].to(device)
            true_output = batch[1].float().to(device)
        
            pred_output = model(input_ids)
            loss = criterion(pred_output, true_output)

            total_val_loss += loss.item()

            _, predicted = torch.max(pred_output.data, 1)
            _, label = torch.max(true_output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print(predicted)
            print(label)

        avg_val_loss = total_val_loss / len(valid_data)
        accuracy = correct / total

        print(f'Epoch {epoch + 1}/{num_epoch}: Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
