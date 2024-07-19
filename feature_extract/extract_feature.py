from transformers import AutoModel
import torch
import torchtext.vocab as tvocab
import numpy as np
import math
from feature_extract.vocabulary import Vocabulary

from constant import PHOBERT_VER, PHOBERT_BATCH_SIZE, MAX_LEN

MODE = ['detail', 'total']
MODEL = ['PhoBERT', 'Phow2v']

def extractFeature(device, ids, attentions=[], mode='detail', model='PhoBERT'):
    if model not in MODEL:
        raise Exception(f'No model named {model}')

    if model == MODEL[0]:
        return usingPhoBERT(device, ids, attentions, mode)
    else:
        return usingPhow2v(device, ids)

def padding(array):
    if len(array) == MAX_LEN:
        return array
    if len(array) > MAX_LEN:
        array = array[0:MAX_LEN]
        return array
    return array + [torch.zeros(300) for _ in range(MAX_LEN - len(array))]

def getWordEmbedding(word_embedding, corpus):
    res = []
    for word_arr in corpus:
        temp = []
        for word in word_arr:
            if word in word_embedding.stoi:
                temp.append(word_embedding.vectors[word_embedding.stoi[word]])
            else:
                temp.append(torch.rand(300))

        temp = padding(temp)
        temp = torch.stack(temp)
        res.append(temp)

    res = torch.stack(res)
    return res

def usingPhow2v(device, texts):
    print('EXTRACTING FEATURE FROM PHOW2V')
    word_embedding = tvocab.Vectors(name='tools/phow2v_300.txt', unk_init=torch.Tensor.normal_)
    print(f'Phow2v original shape: {word_embedding.vectors.shape}')
    
    vocab = Vocabulary()
    vocab_list = list(word_embedding.stoi.keys())
    for word in vocab_list:
        vocab.add(word)

    tensor = vocab.corpus_to_tensor(texts)
    corpus = vocab.tensor_to_corpus(tensor)

    res = getWordEmbedding(word_embedding, corpus)
    print(f'Feature shape: {res.size()}')
    
    return res

def usingPhoBERT(device, ids, attentions, mode='detail'):
    if mode not in MODE:
        raise Exception(f'No mode named {mode}')

    print(f'EXTRACTING FEATURE FROM PHOBERT using {mode} mode')

    phobert = AutoModel.from_pretrained(PHOBERT_VER, output_hidden_states=(True if mode == MODE[1] else False))
    phobert.eval()
    phobert = phobert.to(device)

    num_batch = math.ceil(len(ids) / PHOBERT_BATCH_SIZE)
    final_feature = []
    for size in range(num_batch):
        batch_id = torch.tensor(ids[PHOBERT_BATCH_SIZE * size : min(PHOBERT_BATCH_SIZE * (size + 1), len(ids))]).to(device)
        batch_attention = torch.tensor(attentions[PHOBERT_BATCH_SIZE * size : min(PHOBERT_BATCH_SIZE * (size + 1), len(ids))]).to(device)
        
        with torch.no_grad():
            output = phobert(input_ids=batch_id, attention_mask=batch_attention)

        if mode == MODE[1]:
            res_emb = torch.cat((output[2][-1][:, 0, :], output[2][-2][:, 0, :], output[2][-3][:, 0, :], output[2][-4][:, 0, :]), dim=-1)
        else:
            res_emb = ouput.last_hidden_state

        final_feature.append(res_emb)

        print(f'Batch {size + 1} has done!, Shape: {res_emb.size()}')

    final_feature = torch.cat(final_feature, dim=0)

    print(f'Features shape: {final_feature.size()}')

    return final_feature
