import pandas as pd

AIVIVN_train = pd.read_csv('res/AIVIVN_train_emoji.csv')
UIT_VSFC_train = pd.read_csv('res/UIT_ViHSD_train_emoji.csv')
AIVIVN_test = pd.read_csv('res/AIVIVN_test_emoji.csv')
UIT_VSFC_test = pd.read_csv('res/UIT_ViHSD_test_emoji.csv')

train = pd.concat([AIVIVN_train, UIT_VSFC_train], ignore_index=True)
test = pd.concat([AIVIVN_test, UIT_VSFC_test], ignore_index=True)

train.to_csv('res/benchmark_train_emoji.csv', index=False)
test.to_csv('res/benchmark_test_emoji.csv', index=False)


