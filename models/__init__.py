import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import queue
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from models.BiLSTM import BiLSTM
from models.CNN import CNN2d
from models.XGBoost import XGBoost
from models.LR import LR
from models.GRU import GRU
from models.BiGRU import BiGRU
from models.Attention_BiLSTM import AttentionBiLSTM
from models.CNN_Trans_Enc import CNN_Trans_Enc
from models.BiGRU_CNN_Trans_Enc import BiGRU_CNN_Trans_Enc
from constant import *

with open('models/global_config.json', 'r') as file:
    config = json.load(file)

batch_size = config.get('batch_size')
num_epoch = config.get('num_epoch')

def startTraining(device, model_name, dataset, extract_model):
    train_content = np.load(f'res/features/{extract_model}_{dataset}_train_features.npy')
    test_content = np.load(f'res/features/{extract_model}_{dataset}_test_features.npy')

    train_data = pd.read_csv(f'{DATASET_PATH}/{dataset}_train_emoji.csv')[:100]
    test_data = pd.read_csv(f'{DATASET_PATH}/{dataset}_test_emoji.csv')[:100]

    # mapping = {'neg': 0, 'neu': 1, 'pos': 2}

    train_rating = train_data['label'].apply(int)
    test_rating = test_data['label'].apply(int)

    num_classes = 3 if dataset != AIVIVN else 2

    train_content = torch.tensor(train_content)
    train_rating = torch.tensor(train_rating)
    train_rating = torch.nn.functional.one_hot(train_rating, num_classes=num_classes)

    test_content = torch.tensor(test_content)
    test_rating = torch.tensor(test_rating)
    test_rating = torch.nn.functional.one_hot(test_rating, num_classes=num_classes)

    print('Loading Done')
    print(f'Content Shape: {train_content.size()}')
    print(f'Label Shape: {train_rating.size()}')

    train_data = train_content
    test_data = test_content

    input_shape = train_data.size()

    if model_name == BILSTM:
        train(BiLSTM(device=device, dropout=0.1, input_shape=input_shape, num_classes=num_classes), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)      
    elif model_name == XGBOOST:
        train(XGBoost(), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)      
    elif model_name == LR:
        train(LR(), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)
    elif model_name == GRU:
        train(GRU(device=device, input_shape=input_shape, dropout=0.1, num_classes=num_classes), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)
    elif model_name == BiGRU:
        train(BiGRU(device=device, input_shape=input_shape, dropout=0.1, num_classes=num_classes), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)
    elif model_name == CNN:
        train(CNN2d(device=device, input_shape=input_shape, dropout=0.1, num_classes=num_classes), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)
    elif model_name == ATTENTION_BILSTM:
        train(AttentionBiLSTM(device=device, dropout=0.1, input_shape=input_shape, num_classes=num_classes), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)
    elif model_name == CNN_TRANS_ENCODER:
        train(CNN_Trans_Enc(input_shape=input_shape, dropout=0.1, num_classes=num_classes), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)
    elif model_name == BI_GRU_CNN_TRANS_ENCODER:
        train(BiGRU_CNN_Trans_Enc(input_shape=input_shape, device=device, dropout=0.1, num_classes=num_classes), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, extract_model=extract_model)

def train(model, train_input, train_output, test_input, test_output, device, extract_model):
    model.to(device)
    ML_model = ['XGBoost', 'Logistic_Regression', 'SVM']

    # Splitting dataset
    train_size = int(0.9*train_input.size(0))

    if model.model_name in ML_model:
        _, train_output = torch.max(train_output, 1)
        _, test_output = torch.max(test_output, 1)
        train_output = train_output.numpy()
        test_output = test_output.numpy()
        train_data = train_input.cpu().numpy()
        train_label = train_output
        test_data = test_input.cpu().numpy()
        test_label = test_output

        print("Training")
        model.train()
        model(train_data, train_label)

        model.eval()
        pred = model(x=test_data, train=False)

        report = classification_report(test_label, pred)
        print(report)

    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = opt.Adam(model.parameters(), lr=0.000008)
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5, verbose=True)
        train_data = DataLoader(TensorDataset(train_input[:train_size], train_output[:train_size]), batch_size=batch_size, shuffle=True)
        valid_data = DataLoader(TensorDataset(train_input[train_size:], train_output[train_size:]), batch_size=batch_size, shuffle=True)
        test_data = DataLoader(TensorDataset(test_input, test_output), batch_size=batch_size, shuffle=True)

        best_acc = 0
        best_loss = 10000
        val_train_history = {
            'accuracy': [],
            'epoch': []
        }

        for epoch in range(num_epoch):
            train_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epoch}:")
            val_bar = tqdm(valid_data, desc=f"Epoch {epoch + 1}/{num_epoch}:")

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

            avg_val_loss = total_val_loss / len(valid_data)
            accuracy = correct / total

            print(f'Epoch {epoch + 1}/{num_epoch}: Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

            val_train_history['accuracy'].append(accuracy)
            val_train_history['epoch'].append(epoch + 1)

            # Compare current model with previous model
            if best_acc < accuracy or (best_acc == accuracy and best_loss > avg_val_loss):
                torch.save(model.state_dict(), f'res/models/{extract_model}/{model.model_name}.pth')
                best_acc = accuracy
                best_loss = avg_val_loss
                print('Model saved, current accuracy:', best_acc)

            scheduler.step(avg_val_loss)
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Test model performance
        test_bar = tqdm(test_data, desc=f"Epoch {epoch + 1}/{num_epoch}:")
        model.load_state_dict(torch.load(f'res/models/{extract_model}/{model.model_name}.pth'))
        torch.save(model.state_dict(), f'{DRIVE_PATH}/models/{extract_model}/{model.model_name}.pth')
        model.eval()

        predicted = []
        label = []
        for batch in test_bar:
            input_ids = batch[0].to(device)
            true_output = batch[1].float().to(device)
        
            pred_output = model(input_ids)
            loss = criterion(pred_output, true_output)

            total_val_loss += loss.item()

            _, temp_predicted = torch.max(pred_output.data, 1)
            _, temp_label = torch.max(true_output, 1)
            temp_predicted = temp_predicted.cpu().numpy()
            temp_label = temp_label.cpu().numpy()

            predicted.append(temp_predicted)
            label.append(temp_label)

        predicted = np.array(predicted[0])
        label = np.array(label[0])
        report = classification_report(label, predicted)
        print(report)

    # Save report
    with open(f'{DRIVE_PATH}/report/{extract_model}/{model.model_name}.txt', 'w') as file:
        file.write(report)
    print(f'REPORT saved - {DRIVE_PATH}/report/{extract_model}/{model.model_name}.txt')

    # Visualize model val train processing
    plt.figure()
    plt.plot(val_train_history['accuracy'], label='Valid Accuracy')
    plt.title(f'{model.model_name} Training Process')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'{DRIVE_PATH}/train_process/{extract_model}/{model.model_name}.png')
    plt.close()

    print(f'Image saved - {DRIVE_PATH}/train_process/{extract_model}/{model.model_name}.png')
