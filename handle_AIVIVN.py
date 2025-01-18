import pandas as pd

data1 = pd.read_csv("res/AIVIVN_test.csv")
data2 = pd.read_csv("res/AIVIVN_train.csv")

data1['label'] = data1['label'].apply(lambda x: 2 if x == 1 else 0)
data2['label'] = data2['label'].apply(lambda x: 2 if x == 1 else 0)

data1.to_csv("res/AIVIVN_test.csv", index=False)
data2.to_csv("res/AIVIVN_train.csv", index=False)
