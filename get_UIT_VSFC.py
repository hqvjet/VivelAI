from datasets import load_dataset
import pandas as pd

ds = load_dataset("uitnlp/vietnamese_students_feedback")

train = pd.DataFrame(ds['train'])
valid = pd.DataFrame(ds['validation'])

train = pd.concat([train, valid], ignore_index=True)
test = pd.DataFrame(ds['test'])
train.rename(columns={'sentiment': 'label', 'sentence': 'comment'}, inplace=True)
test.rename(columns={'sentiment': 'label', 'sentence': 'comment'}, inplace=True)
train = train.drop(columns=['topic'])
test = test.drop(columns=['topic'])

train.to_csv('res/UIT_VSFC_train.csv', index=False)
test.to_csv('res/UIT_VSFC_test.csv', index=False)

