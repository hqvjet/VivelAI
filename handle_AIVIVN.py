import pandas as pd

data = pd.read_csv("res/AIVIVN_test.csv")

data['label'] = data['label'].apply(lambda x: 2 if x == 1 else 0)

data.to_csv("res/AIVIVN_test.csv", index=False)
