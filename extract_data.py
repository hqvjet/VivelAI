import pandas as pd

def five_to_three(rating):
    if rating in [1, 2]:
        return 1
    elif rating == 3:
        return 2
    else:
        return 3

def text_to_num(rating):
    if rating == 'NEG':
        return 1
    elif rating == 'NEU':
        return 2
    else:
        return 3

data_1 = pd.read_csv('res/data_service.csv')
data_2 = pd.read_csv('res/neu_n_neg.csv')
data_3 = pd.read_csv('res/neg.csv') # Full negative comment

data_1 = data_1.drop(columns=['placeInfo/name'])
data_2 = data_2.drop(columns=['rating'])
data_2 = data_2.rename(columns={'label': 'rating'})
data_3['rating'] = 1

data_1['rating'] = data_1['rating'].apply(five_to_three)
data_2['rating'] = data_2['rating'].apply(text_to_num)

# data 1
pos = data_1[data_1['rating'] == 3]
pos = pos.head(8000)

remain = data_1[data_1['rating'] < 3]
#
# res = pd.concat([pos, remain])
# res = pd.concat([res, data_2])

res = pd.concat([pos, remain, data_2, data_3])
res = res.sample(frac=1).reset_index(drop=True)

print(res)

res.to_csv('res/true_data.csv', index=False)
