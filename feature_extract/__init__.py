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
    titles = data['title'].apply(str)
    contents = data['content'].apply(str)

    return titles, contents
        
def useFeatureExtractor(device):
    train_data = getDataset('res/train_emoji.csv')
    test_data = getDataset('res/test_emoji.csv')
    train_title, train_content = makeData(train_data)
    test_title, test_content = makeData(test_data)

    train_title = useNormalize(train_title)
    train_content = useNormalize(train_content)
    test_title = useNormalize(test_title)
    test_content = useNormalize(test_content)

    train_title = useLemma(train_title)
    train_content = useLemma(train_content)
    test_title = useLemma(test_title)
    test_content = useLemma(test_content)

    train_title = useTokenize(train_title)
    train_content = useTokenize(train_content)
    test_title = useTokenize(test_title)
    test_content = useTokenize(test_content)

    train_title = removeStopword(train_title)
    train_content = removeStopword(train_content)
    test_title = removeStopword(test_title)
    test_content = removeStopword(test_content)

    # train_title = emojiHandling(train_title)
    # train_content = emojiHandling(train_content)
    # test_title = emojiHandling(test_title)
    # test_content = emojiHandling(test_content)

    # e_matrix = getEmojiEmbeddingMatrix()
 
    key = input('Choose feature extractor method:\n1. PhoBERT\n2. PhoW2V\nYour Input: ')
    
    if key == '1':
        model = 'phobert'
    elif key == '2':
        model = 'phow2v'
    else:
        print('Wrong method, please try again')

    if key == '1':
        tokenizer = getTokenizer()
        train_title, train_title_attention = useIdentify(train_title, tokenizer)
        train_content, train_content_attention = useIdentify(train_content, tokenizer)
        test_title, test_title_attention = useIdentify(test_title, tokenizer)
        test_content, test_content_attention = useIdentify(test_content, tokenizer)

        # train_title = extractFeature(device, train_title, train_title_attention, model=model, tokenizer=tokenizer, emoji_matrix=e_matrix)
        # train_content = extractFeature(device, train_content, train_content_attention, model=model, tokenizer=tokenizer, emoji_matrix=e_matrix)
        # test_title = extractFeature(device, test_title, test_title_attention, model=model, tokenizer=tokenizer, emoji_matrix=e_matrix)
        # test_content = extractFeature(device, test_content, test_content_attention, model=model, tokenizer=tokenizer, emoji_matrix=e_matrix)

        train_title = extractFeature(device, train_title, train_title_attention, model=model, tokenizer=tokenizer)
        train_content = extractFeature(device, train_content, train_content_attention, model=model, tokenizer=tokenizer)
        test_title = extractFeature(device, test_title, test_title_attention, model=model, tokenizer=tokenizer)
        test_content = extractFeature(device, test_content, test_content_attention, model=model, tokenizer=tokenizer)

    else:
        train_title = extractFeature(device, train_title, model=model)
        train_content = extractFeature(device, train_content, model=model)
        test_title = extractFeature(device, test_title, model=model)
        test_content = extractFeature(device, test_content, model=model)

    np.save(f'res/features/{model}_train_title_features_icon.npy', train_title.cpu())
    np.save(f'res/features/{model}_train_content_features_icon.npy', train_content.cpu())
    np.save(f'res/features/{model}_test_title_features_icon.npy', test_title.cpu())
    np.save(f'res/features/{model}_test_content_features_icon.npy', test_content.cpu())
