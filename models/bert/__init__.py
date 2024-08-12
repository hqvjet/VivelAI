import math
import torch
import time
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from models.bert.bert import BERT
from models.bert.config import num_layer, num_head, batch_size, dropout, learning_rate, num_epoch

def trainBERT(device, title, content, label):
    # title = title[0:256]
    # content = content[0:256]
    # label = label[0:256]

    train = int(0.8 * title.size(0))
    test = title.size(0) - train
    temp = train
    train = int(train * 0.9)
    val = temp - train
    del temp

    train_title = title[0:train]   
    val_title = title[train:train+val]
    test_title = title[train+val:train+val+test]

    train_content = content[0:train]
    val_content = content[train:train+val]
    test_content = content[train+val:train+val+test]

    train_label = label[0:train]
    val_label = label[train:train+val]
    test_label = label[train+val:train+val+test]

    train_dataloader = DataLoader(TensorDataset(torch.cat((train_title, train_content), dim=-1), train_label), num_workers=6, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(torch.cat((val_title, val_content), dim=-1), val_label), num_workers=6, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(torch.cat((test_title, test_content), dim=-1), test_label), num_workers=6, batch_size=batch_size, shuffle=True)

    model = BERT(
        num_head=num_head, max_len=title.size(1), debug=True,
        dropout=dropout, device=device, d_model=title.size(2)*2,
        num_layer=24
    )
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=learning_rate)
    weights = torch.tensor([0.2586, 1.0, 0.3543]).to(device)
    crite = CrossEntropyLoss(weight=weights)
    
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epoch}'):
            input_ids = batch[0].to(device)
            label = batch[1].float().to(device)

            opt.zero_grad()
            output = model(input_ids)
            loss = crite(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Validation'):
                input_ids = batch[0].to(device)
                label = batch[1].to(device).float()

                output = model(input_ids)
                loss = crite(output, label)
                total_val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                _, label = torch.max(label, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct / total

        print(f'Epoch {epoch + 1}/{num_epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

    torch.save(model, 'res/models/BERT.pth')
