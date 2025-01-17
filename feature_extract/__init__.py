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

def getDataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as error:
        print('ERROR WHILE READING DATASET')
        print(error)

def makeData(data):
    # titles = data['title'].apply(str)
    contents = data['comment'].apply(str)

    # return titles, contents
    return contents
        
def useFeatureExtractor(device):
    train_data = getDataset('res/benchmark_train_emoji.csv')
    test_data = getDataset('res/benchmark_test_emoji.csv')
    train_content = makeData(train_data)
    test_content = makeData(test_data)

    train_content = useNormalize(train_content)
    test_content = useNormalize(test_content)

    train_content = useLemma(train_content)
    test_content = useLemma(test_content)

    train_content = useTokenize(train_content)
    test_content = useTokenize(test_content)

    train_content = removeStopword(train_content)
    test_content = removeStopword(test_content)

    print(train_content[:10])

    key = input('Choose feature extractor method:\n1. PhoBERT\n2. PhoW2V\nYour Input: ')
    
    if key == '1':
        model = 'phobert'
    elif key == '2':
        model = 'phow2v'
    else:
        print('Wrong method, please try again')

    if key == '1':
        tokenizer = getTokenizer()
        train_content, train_content_attention = useIdentify(train_content, tokenizer)
        test_content, test_content_attention = useIdentify(test_content, tokenizer)

        train_content = extractFeature(device, train_content, train_content_attention, model=model, tokenizer=tokenizer)
        test_content = extractFeature(device, test_content, test_content_attention, model=model, tokenizer=tokenizer)

    else:
        train_content = extractFeature(device, train_content, model=model)
        test_content = extractFeature(device, test_content, model=model)

    np.save(f'res/features/{model}_train_content_features_viso.npy', train_content.cpu())
    np.save(f'res/features/{model}_test_content_features_viso.npy', test_content.cpu())
