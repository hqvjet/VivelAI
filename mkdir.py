import os
from constant import *

PRE = 'res/models'

if not os.path.exists(PRE + '/' + BILSTM):
    os.makedirs(PRE + '/' + BILSTM)
else:
    print('Folder exists: ' + PRE + '/' + BILSTM)

if not os.path.exists(PRE + '/' + XGBOOST):
    os.makedirs(PRE + '/' + XGBOOST)
else:
    print('Folder exists: ' + PRE + '/' + XGBOOST)

if not os.path.exists(PRE + '/' + LR):
    os.makedirs(PRE + '/' + LR)
else:
    print('Folder exists: ' + PRE + '/' + LR)

if not os.path.exists(PRE + '/' + GRU):
    os.makedirs(PRE + '/' + GRU)
else:
    print('Folder exists: ' + PRE + '/' + GRU)

if not os.path.exists(PRE + '/' + BIGRU):
    os.makedirs(PRE + '/' + BIGRU)
else:
    print('Folder exists: ' + PRE + '/' + BIGRU)

if not os.path.exists(PRE + '/' + CNN):
    os.makedirs(PRE + '/' + CNN)
else:
    print('Folder exists: ' + PRE + '/' + CNN)

if not os.path.exists(PRE + '/' + ATTENTION_BILSTM):
    os.makedirs(PRE + '/' + ATTENTION_BILSTM)
else:
    print('Folder exists: ' + PRE + '/' + ATTENTION_BILSTM)

if not os.path.exists(PRE + '/' + CNN_TRANS_ENC):
    os.makedirs(PRE + '/' + CNN_TRANS_ENC)
else:
    print('Folder exists: ' + PRE + '/' + CNN_TRANS_ENC)

if not os.path.exists(PRE + '/' + BIGRU_CNN_TRANS_ENC):
    os.makedirs(PRE + '/' + BIGRU_CNN_TRANS_ENC)
else:
    print('Folder exists: ' + PRE + '/' + BIGRU_CNN_TRANS_ENC)



