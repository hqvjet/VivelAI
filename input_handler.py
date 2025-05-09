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
from models.GRU import GRU

if torch.cuda.is_available():
    print('USING GPU')
    device = torch.device('cuda')
else:
    print('USING CPU')
    device = torch.device('cpu')
    
emoji_size = torch.Size((1, 3072))

E2V_model = BiGRU(device=device, input_shape=emoji_size, num_classes=2)
E2V_model = E2V_model.to(device)
E2V_model.load_state_dict(torch.load('res/models/E2V-PHOBERT/AIVIVN/BiGRU.pth'))
E2V_model.eval()
E2T_model = GRU(device=device, input_shape=emoji_size, num_classes=2)
E2T_model = E2T_model.to(device)
E2T_model.load_state_dict(torch.load('res/models/E2T-PHOBERT/AIVIVN/GRU.pth'))
E2T_model.eval()

e_matrix = getEmojiEmbeddingMatrix()
E2V_tokenizer = getTokenizer(E2V_PHOBERT, e_matrix)
E2T_tokenizer = getTokenizer(E2T_PHOBERT)

def handle_input_using_E2VPhoBERT(input: Comment):
    comment = pd.Series([input.comment])

    comment = useNormalize(comment, E2V_PHOBERT)
    comment = useLemma(comment)
    comment = useTokenize(comment)
    comment = removeStopword(comment)
    comment = emojiHandling(comment, mode='unicode') 
    comment, comment_attention = useIdentify(comment, E2V_tokenizer)
    comment = extractFeature(device, comment, comment_attention, extract_model=E2V_PHOBERT, tokenizer=E2V_tokenizer, emoji_matrix=e_matrix)

    inp = comment.to(device)
    pred = E2V_model(inp)
    _, pred = torch.max(pred.data, 1)

    return pred.cpu().tolist()

def handle_input_using_E2TPhoBERT(input: Comment):
    comment = pd.Series([input.comment])

    comment = useNormalize(comment, E2T_PHOBERT)
    comment = useLemma(comment)
    comment = useTokenize(comment)
    comment = removeStopword(comment)
    comment = emojiHandling(comment, mode='text') 
    comment, comment_attention = useIdentify(comment, E2T_tokenizer)
    comment = extractFeature(device, comment, comment_attention, extract_model=E2T_PHOBERT, tokenizer=E2T_tokenizer, emoji_matrix=e_matrix)

    inp = comment.to(device)
    pred = E2T_model(inp)
    _, pred = torch.max(pred.data, 1)

    return pred.cpu().tolist()
