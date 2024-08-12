import pandas as pd
import numpy as np

from feature_extract.normalize import useNormalize
from feature_extract.lemma import useLemma
from feature_extract.tokenize import useTokenize
from feature_extract.remove_stopword import removeStopword
from feature_extract.identify import useIdentify
from feature_extract.extract_feature import extractFeature

def getDataset(file_path):
    try:
        # return pd.read_excel(file_path, engine='openpyxl')
        return pd.read_csv(file_path)
    except Exception as error:
        print('ERROR WHILE READING DATASET')
        print(error)

def makeData(data):
    titles = data['Title'][:8804].apply(str)
    contents = data['Content'][:8804].apply(str)

    return titles, contents
        
def useFeatureExtractor(device):
    data = getDataset('res/data.csv')
    title, content = makeData(data)

    title = useNormalize(title)
    content = useNormalize(content)

    title = useLemma(title)
    content = useLemma(content)

    title = useTokenize(title)
    content = useTokenize(content)
 
    key = input('Choose feature extractor method:\n1. PhoBERT\n2. PhoW2V\nYour Input: ')
    
    if key == '1':
        model = 'PhoBERT'
        key1 = input('Choose PhoBERT type:\n1. Total\n2. Detail\nYour Input: ')
        if key1 == '1':
            mode = 'total'
        elif key1 == '2':
            mode = 'detail'
        else:
            print('Wrong method, please try again')

    elif key == '2':
        model = 'Phow2v'
    else:
        print('Wrong method, please try again')

    if key == 1:
        title, title_attention = useIdentify(title)
        content, content_attention = useIdentify(content)

    else:
        mode=None

    title = extractFeature(device, title, mode=mode, model=model)
    content = extractFeature(device, content, mode=mode, model=model)

    if key == 1 and key1 == 1:
        name = 'phobert_total'
    elif key == 1 and key1 == 2:
        name = 'phobet_detail'
    else:
        name = 'phow2v'
    np.save(f'res/features/{name}_title_features.npy', title.cpu())
    np.save(f'res/features/{name}_content_features.npy', content.cpu())
