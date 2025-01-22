from transformers import AutoModel
import torch
import torchtext.vocab as tvocab
import numpy as np
import math
from constant import *

def extractFeature(device, ids, attentions, extract_model, tokenizer, emoji_matrix):
    phobert = AutoModel.from_pretrained(PHOBERT_VER if extract_model != VISOBERT else VISOBERT_VER, output_hidden_states=True)
    phobert.eval()

    if emoji_matrix != None:
        print('AAAAAAAAAAAAAAAAAAAAAA')
        phobert.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            for token in e_matrix.keys():
                token_id = tokenizer.convert_tokens_to_ids(token)
                phobert.embeddings.word_embeddings.weight[token_id] = e_matrix[token]

    phobert = phobert.to(device)

    num_batch = math.ceil(len(ids) / EXTRACT_BATCH_SIZE)
    final_feature = []
    for size in range(num_batch):
        batch_id = torch.tensor(ids[EXTRACT_BATCH_SIZE * size : min(EXTRACT_BATCH_SIZE * (size + 1), len(ids))]).to(device)
        batch_attention = torch.tensor(attentions[EXTRACT_BATCH_SIZE * size : min(EXTRACT_BATCH_SIZE * (size + 1), len(ids))]).to(device)
        print(batch_id.size(), batch_attention.size())
        
        with torch.no_grad():
            output = phobert(input_ids=batch_id, attention_mask=batch_attention)

        res_emb = torch.cat((output[2][-1][:, 0, :], output[2][-2][:, 0, :], output[2][-3][:, 0, :], output[2][-4][:, 0, :]), dim=-1)
        final_feature.append(res_emb)

        print(f'Batch {size + 1} has done!, Shape: {res_emb.size()}')

    final_feature = torch.cat(final_feature, dim=0)
    print(final_feature, final_feature.size())

    print(f'Features shape: {final_feature.size()}')

    return final_feature
