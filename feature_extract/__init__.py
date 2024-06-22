import pandas as pd
import numpy as np
from vncorenlp import VnCoreNLP

# INITIALIZE VAR
rdr = VnCoreNLP("tools/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

def getDataset(file_path):
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except:
        pass

def makeData(data):
    titles = data['title'].apply(str)
    contents = data['content'].apply(str)
    ratings = data['rating']
    
    return titles, contents, ratings

def tokenizeTexts(texts):
    res = []
    for text in texts:
        temp = []
        for sentence in rdr.tokenize(text):
            temp.append(' '.join(sentence))
        res.append('. '.join(temp))

    print(res)

    return np.array(res)

def normalize(texts):
    for m in range(len(texts)):
        print(texts[m])
        tokens = texts[m].strip().split()
        temp = []
        for i in range(len(tokens)):
            pre_token = ''
            j = 0
            while(j < len(tokens[i])):
                if pre_token == tokens[i][j]:
                    tokens[i] = tokens[i][0 : j] + tokens[i][min(len(tokens[i]), j + 1):]
                    j-=1
                else:
                    pre_token = tokens[i][j]
                j += 1

            temp.append(tokens[i])
        texts[m] = ' '.join(temp)
    return texts
        
def useFeatureExtractor():
    data = getDataset('res/dataset.xlsx')
    title, content, rating = makeData(data)

    title = normalize(title)
    content = normalize(content)
    title = tokenizeTexts(title)
    content = tokenizeTexts(content)
    

