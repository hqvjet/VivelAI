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

from models.LSTM import LSTM
from models.CNN_BILSTM import CNNnBiLSTM

if torch.cuda.is_available():
    print('USING GPU')
    device = torch.device('cuda')
else:
    print('USING CPU')
    device = torch.device('cpu')
    
title_size = torch.Size((1, 6144))
comment_size = torch.Size((1, 3072))

title_model = LSTM(device, emb_tech=1, dropout=0.3, input_shape=title_size)
title_model = title_model.to(device)
title_model.load_state_dict(torch.load('res/models/with_title/phobert/LSTM.pth'))
title_model.eval()

comment_model = CNNnBiLSTM(device, input_shape=comment_size, emb_tech=1, useTitle=False)
comment_model = comment_model.to(device)
comment_model.load_state_dict(torch.load('res/models/no_title/phobert/Ensemble_CNN_BiLSTM.pth'))
comment_model.eval()

def handle_input_with_title(input: Title_Comment):
    title = pd.Series([input.title])
    comment = pd.Series([input.comment])

    title = useNormalize(title)
    comment = useNormalize(comment)

    title = useLemma(title)
    comment = useLemma(comment)

    title = useTokenize(title)
    comment = useTokenize(comment)

    title, title_attention = useIdentify(title)
    comment, comment_attention = useIdentify(comment)

    title = extractFeature(device, title, title_attention)
    comment = extractFeature(device, comment, comment_attention)

    inp = torch.cat((title, comment), dim=-1)
    inp = inp.to(device)
    pred = title_model(inp)
    _, pred = torch.max(pred.data, 1)

    return pred.cpu().tolist()

def handle_input_no_title(input: Comment):
    comment = pd.Series([input.comment])

    comment = useNormalize(comment)

    comment = useLemma(comment)

    comment = useTokenize(comment)

    comment, comment_attention = useIdentify(comment)

    comment = extractFeature(device, comment, comment_attention)

    inp = comment.to(device)
    pred = comment_model(inp)
    _, pred = torch.max(pred.data, 1)

    return pred.cpu().tolist()
