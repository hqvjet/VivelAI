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
from models.LSTM import LSTM
from models.BiLSTM import BiLSTM
from models.CNN import CNN2d
from models.XGBoost import XGBoost
from models.FullyConnected import FC
from models.LR import LR
from models.CNN_LSTM import CNNnLSTM
from models.CNN_BILSTM import CNNnBiLSTM
from models.Transformer import Transformer
from models.GRU import GRU
from models.BiGRU import BiGRU
from models.Attention_BiLSTM import AttentionBiLSTM
from constant import DRIVE_PATH, DATASET_PATH

with open('models/global_config.json', 'r') as file:
    config = json.load(file)

batch_size = config.get('batch_size')
num_epoch = config.get('num_epoch')

def startTraining(device):
    key = input('Choose feature source:\n1. PhoBERT\n2. PhoW2V\nYour Input: ')

    print('Loading extracted feature')

    if key == '1':
        source = 'phobert'
    elif key == '2':
        source = 'phow2v'
    else:
        print('Wrong source, please try again')

    train_content = np.load(f'res/features/{source}_train_content_features_icon.npy')
    test_content = np.load(f'res/features/{source}_test_content_features_icon.npy')

    train_data = pd.read_csv(f'{DATASET_PATH}/UIT_VSFC_train_emoji.csv')
    test_data = pd.read_csv(f'{DATASET_PATH}/UIT_VSFC_test_emoji.csv')

    # mapping = {'neg': 0, 'neu': 1, 'pos': 2}

    train_rating = train_data['label'].apply(int)
    test_rating = test_data['label'].apply(int)

    train_content = torch.tensor(train_content)
    train_rating = torch.tensor(train_rating)
    train_rating = torch.nn.functional.one_hot(train_rating, num_classes=3)

    test_content = torch.tensor(test_content)
    test_rating = torch.tensor(test_rating)
    test_rating = torch.nn.functional.one_hot(test_rating, num_classes=3)

    print('Loading Done')
    print(f'Content Shape: {train_content.size()}')
    print(f'Label Shape: {train_rating.size()}')

    key = input('Use Title and Content ?\n1. Yes\n2. No\nYour Input: ')
    if key == '1':
        train_data = train_content
        test_data = test_content
    else:
        train_data = train_content
        test_data = test_content
    useTitle = False if key == '2' else True

    key = input('Choose one of these classification to train:\n1. LSTM\n2. BiLSTM\n3. XGBoost\n4. LG\n5. Ensemble CNN LSTM\n6. Ensemble CNN BiLSTM\n7. GRU\n8. BiGRU\n9. Transformer\n10. CNN\nYour Input: ')
    emb_tech = 1 if source == 'phobert' else 2
    input_shape = train_data.size()

    if key == '1':
        train(LSTM(device=device, dropout=0.3, emb_tech=emb_tech, input_shape=input_shape), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)
    elif key == '2':
        train(BiLSTM(device=device, dropout=0.1, emb_tech=emb_tech, input_shape=input_shape), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)      
    elif key == '3':
        train(XGBoost(emb_tech=emb_tech, useTitle=useTitle), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)      
    elif key == '4':
        train(LR(emb_tech=emb_tech, useTitle=useTitle), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)
    elif key == '5':
        train(CNNnLSTM(device=device, input_shape=input_shape, useTitle=useTitle, emb_tech=emb_tech, dropout=0.1), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)
    elif key == '6':
        train(CNNnBiLSTM(device=device, input_shape=input_shape, useTitle=useTitle, emb_tech=emb_tech, dropout=0.1), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)
    elif key == '7':
        train(GRU(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)
    elif key == '8':
        train(BiGRU(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)
    elif key == '9':
        train(Transformer(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)
    elif key == '10':
        train(CNN2d(device=device, input_shape=input_shape, emb_tech=emb_tech, dropout=0.1), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, input=data, output=rating, device=device, useTitle=useTitle)
    elif key == '11':
        train(AttentionBiLSTM(device=device, dropout=0.1, emb_tech=emb_tech, input_shape=input_shape), train_input=train_data, train_output=train_rating, test_input=test_data, test_output=test_rating, device=device, useTitle=useTitle)      
    else:
        print('Wrong key of model, please choose again')

def train(model, train_input, train_output, test_input, test_output, device, useTitle):
    model.to(device)
    direction = 'with_title' if useTitle else 'no_title'
    model_direction = 'phobert' if model.emb_tech == 1 else 'phow2v'
    ML_model = ['XGBoost', 'Logistic_Regression']

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
        optimizer = opt.Adam(model.parameters(), lr=0.00001)
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
                torch.save(model.state_dict(), f'res/models/{direction}/{model_direction}/{model.model_name}_icon.pth')
                best_acc = accuracy
                best_loss = avg_val_loss
                print('Model saved, current accuracy:', best_acc)

            scheduler.step(avg_val_loss)
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Test model performance
        test_bar = tqdm(test_data, desc=f"Epoch {epoch + 1}/{num_epoch}:")
        model.load_state_dict(torch.load(f'res/models/{direction}/{model_direction}/{model.model_name}_icon.pth'))
        torch.save(model.state_dict(), f'{DRIVE_PATH}/models/{direction}/{model_direction}/{model.model_name}_icon.pth')
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
    with open(f'{DRIVE_PATH}/report/{direction}/{model_direction}/{model.model_name}_icon.txt', 'w') as file:
        file.write(report)
    print(f'REPORT saved - {DRIVE_PATH}/report/{direction}/{model_direction}/{model.model_name}_icon.txt')

    # Visualize model val train processing
    plt.figure()
    plt.plot(val_train_history['accuracy'], label='Valid Accuracy')
    plt.title(f'{model.model_name} Training Process')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'{DRIVE_PATH}/train_process/{direction}/{model_direction}/{model.model_name}_icon.png')
    plt.close()

    print(f'Image saved - {DRIVE_PATH}/train_process/{direction}/{model_direction}/{model.model_name}_icon.png')
