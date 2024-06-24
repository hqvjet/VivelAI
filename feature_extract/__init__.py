import pandas as pd
import numpy as np

from feature_extract.normalize import useNormalize
from feature_extract.tokenize import useTokenize
from feature_extract.lemma import useLemma

def getDataset(file_path):
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except:
        print('ERROR WHILE READING DATASET')

def makeData(data):
    titles = data['title'].apply(str)
    contents = data['content'].apply(str)
    ratings = data['rating'].apply(int)
    
    return titles, contents, ratings
        
def useFeatureExtractor():
    data = getDataset('res/datasets.xlsx')
    title, content, rating = makeData(data)

    title = useNormalize(title)
    content = useNormalize(content)

    title = useLemma(title)
    content = useLemma(content)

    title = useTokenize(title)
    content = useTokenize(content)

    np.savetxt('res/title.txt', title, fmt='%s')
    np.savetxt('res/content.txt', content, fmt='%s')
