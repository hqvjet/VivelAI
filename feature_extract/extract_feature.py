from transformers import AutoModel
import torch
import numpy as np
import math

from constant import PHOBERT_VER, PHOBERT_BATCH_SIZE

def extractFeature(device, ids, attentions):
    print('EXTRACTING FEATURE FROM PHOBERT')

    phobert = AutoModel.from_pretrained(PHOBERT_VER, output_hidden_states=True)
    phobert.eval()
    phobert = phobert.to(device)

    num_batch = math.ceil(len(ids) / PHOBERT_BATCH_SIZE)
    final_cls_feature = []
    for size in range(num_batch):
        batch_id = torch.tensor(ids[PHOBERT_BATCH_SIZE * size : min(PHOBERT_BATCH_SIZE * (size + 1), len(ids))]).to(device)
        batch_attention = torch.tensor(attentions[PHOBERT_BATCH_SIZE * size : min(PHOBERT_BATCH_SIZE * (size + 1), len(ids))]).to(device)
        
        with torch.no_grad():
            output = phobert(input_ids=batch_id, attention_mask=batch_attention)
        cls_emb = torch.cat((output[2][-1][:, 0, :], output[2][-2][:, 0, :], output[2][-3][:, 0, :], output[2][-4][:, 0, :]), dim=-1)
        final_cls_feature.append(cls_emb)

        print(f'Batch {size + 1} has done!, Shape: {cls_emb.size()}')

    final_cls_feature = torch.cat(final_cls_feature, dim=0)

    print(f'Features shape: {final_cls_feature.size()}')

    return final_cls_feature
