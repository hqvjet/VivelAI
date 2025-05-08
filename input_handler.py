import pandas as pd
import numpy as np
import torch

from schemas import Title_Comment, Comment
from feature_extract.normalize import useNormalize
from feature_extract.lemma import useLemma
from feature_extract.tokenize import useTokenize
from feature_extract.remove_stopword import removeStopword
from feature_extract.identify import useIdentify
from feature_extract.extract_feature import extractFeature
from feature_extract.emoji_handling import emojiHandling, getEmojiEmbeddingMatrix
from feature_extract.get_tokenizer import getTokenizer
from constant import *

from models.BiGRU import BiGRU

if torch.cuda.is_available():
    print('USING GPU')
    device = torch.device('cuda')
else:
    print('USING CPU')
    device = torch.device('cpu')
    
title_size = torch.Size((1, 6144))
emoji_size = torch.Size((1, 3072))

emoji_model = BiGRU(device=device, input_shape=emoji_size, num_classes=2)
emoji_model = emoji_model.to(device)
emoji_model.load_state_dict(torch.load('res/models/E2V-PHOBERT/AIVIVN/BiGRU.pth'))
emoji_model.eval()
e_matrix = getEmojiEmbeddingMatrix()
tokenizer = getTokenizer(E2V_PHOBERT, e_matrix)

def handle_input_using_E2VPhoBERT(input: Comment, extract_model=None):
    comment = pd.Series([input.comment])

    comment = useNormalize(comment, extract_model)
    comment = useLemma(comment)
    comment = useTokenize(comment)
    comment = removeStopword(comment)
    comment = emojiHandling(comment, mode='unicode') 
    comment, comment_attention = useIdentify(comment, tokenizer)
    comment = extractFeature(device, comment, comment_attention, extract_model=extract_model, tokenizer=tokenizer, emoji_matrix=e_matrix)

    inp = comment.to(device)
    pred = emoji_model(inp)
    _, pred = torch.max(pred.data, 1)

    return pred.cpu().tolist()
