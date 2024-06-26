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
        return pd.read_excel(file_path, engine='openpyxl')
    except:
        print('ERROR WHILE READING DATASET')

def makeData(data):
    titles = data['processed_title'].apply(str)
    contents = data['processed_review'].apply(str)
    ratings = data['user_rate'].apply(int)

    return titles, contents, ratings
        
def useFeatureExtractor(device):
    data = getDataset('res/datasets.xlsx')
    title, content, rating = makeData(data)

    title = useNormalize(title)
    content = useNormalize(content)

    title = useLemma(title)
    content = useLemma(content)

    title = useTokenize(title)
    content = useTokenize(content)

    title, title_attention = useIdentify(title)
    content, content_attention = useIdentify(content)

    title = extractFeature(device, title, title_attention)
    content = extractFeature(device, content, content_attention)

    # Save features
    np.save('res/features/phobert_title_features.npy', title.cpu())
    np.save('res/features/phobert_content_features.npy', content.cpu())

    # np.savetxt('res/title.txt', title, fmt='%s')
    # np.savetxt('res/content.txt', content, fmt='%s')
