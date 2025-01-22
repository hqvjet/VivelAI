import pandas as pd
import numpy as np

from feature_extract.normalize import useNormalize
from feature_extract.lemma import useLemma
from feature_extract.tokenize import useTokenize
from feature_extract.remove_stopword import removeStopword
from feature_extract.identify import useIdentify
from feature_extract.extract_feature import extractFeature
from feature_extract.emoji_handling import emojiHandling, getEmojiEmbeddingMatrix
from feature_extract.get_tokenizer import getTokenizer
from constant import *

def getDataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as error:
        print('ERROR WHILE READING DATASET')
        print(error)

def makeData(data):
    # titles = data['title'].apply(str)
    contents = data['comment'].apply(str)[:100]

    # return titles, contents
    return contents
        
def useFeatureExtractor(device, extract_model, dataset):
    train_data = getDataset(f'{DATASET_PATH}/{dataset}_train_emoji.csv')
    test_data = getDataset(f'{DATASET_PATH}/{dataset}_test_emoji.csv')

    train_content = makeData(train_data)
    test_content = makeData(test_data)

    train_content = useNormalize(train_content, extract_model)
    test_content = useNormalize(test_content, extract_model)

    train_content = useLemma(train_content)
    test_content = useLemma(test_content)

    train_content = useTokenize(train_content)
    test_content = useTokenize(test_content)

    train_content = removeStopword(train_content)
    test_content = removeStopword(test_content)

    if extract_model == E2V_PHOBERT:
        train_content = emojiHandling(train_content)
        test_content = emojiHandling(test_content)

    e_matrix = getEmojiEmbeddingMatrix() if extract_model == E2V_PHOBERT else None

    print(train_content[:10])
    
    tokenizer = getTokenizer(extract_model, e_matrix if extract_model == E2V_PHOBERT else None)
    train_content, train_content_attention = useIdentify(train_content, tokenizer)
    test_content, test_content_attention = useIdentify(test_content, tokenizer)

    train_content = extractFeature(device, train_content, train_content_attention, extract_model=extract_model, tokenizer=tokenizer, emoji_matrix=e_matrix)
    test_content = extractFeature(device, test_content, test_content_attention, extract_model=extract_model, tokenizer=tokenizer, emoji_matrix=e_matrix)

    np.save(f'res/features/{extract_model}_{dataset}_train_features.npy', train_content.cpu())
    np.save(f'res/features/{extract_model}_{dataset}_test_features.npy', test_content.cpu())
