import pandas as pd

AIVIVN_train = pd.read_csv('res/AIVIVN_train.csv')
AIVIVN_train = AIVIVN_train.drop(columns=['id'])
UIT_VSFC_train = pd.read_csv('res/UIT_VSFC_train.csv')
AIVIVN_test = pd.read_csv('res/AIVIVN_test.csv')
AIVIVN_test = AIVIVN_test.drop(columns=['id'])
UIT_VSFC_test = pd.read_csv('res/UIT_VSFC_test.csv')

train = pd.concat([AIVIVN_train, UIT_VSFC_train], ignore_index=True)
test = pd.concat([AIVIVN_test, UIT_VSFC_test], ignore_index=True)

train.to_csv('res/benchmark_train.csv', index=False)
test.to_csv('res/benchmark_test.csv', index=False)


