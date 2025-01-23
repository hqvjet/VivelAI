import subprocess
import time
from constant import *
import os

MODEL_CMD = ['1\n']
DATASET_CMD = ['1\n', '2\n', '3\n']
ACT_CMD = ['1\n', '2\n']
TRAINING_CMD = ['1\n', '2\n', '3\n', '4\n', '5\n', '6\n', '7\n', '8\n', '9\n']

def run(command):
    process = subprocess.Popen(['python', '__init__.py'], stdin=subprocess.PIPE, text=True)
    process.communicate(command)

    if process.returncode != 0:
        print('Process error !')
        exit()
    
    print('Finish 1 automic process')
    print("Reset GPU...")
    os.system("kill -9 $(nvidia-smi | grep 'python' | awk '{print $5}')")
    print('Reset GPU Done !')

print('Do you wanna extract feature ?<y/n>')

key = input()

if key == 'y':
    print('Start Extracting Features')
    for m_c in MODEL_CMD:
        if m_c == '1\n':
            model = E2T_PHOBERT
        elif m_c == '2\n':
            model = PHOBERT
        elif m_c == '3\n':
            model = VISOBERT
        else:
            model = E2V_PHOBERT

        print(f'Start Extracting Features with extraction model {model}')
        for d_c in DATASET_CMD:
            print(f'Extracting Features with extraction model {model} and dataset {d_c}')
            run(m_c + d_c + ACT_CMD[0])

            print(f'Finish Extracting Features with extraction model {model} and dataset {d_c}')
        print(f'Finish Extracting Features with extraction model {model}')

    print('Finish Extracting Features')

for m_c in MODEL_CMD:
    if m_c == '1\n':
        model = E2T_PHOBERT
    elif m_c == '2\n':
        model = PHOBERT
    elif m_c == '3\n':
        model = VISOBERT
    else:
        model = E2V_PHOBERT

    for d_c in DATASET_CMD:
        print(f'Start Training 9 models with dataset {d_c}, extraction model {model}')
        for t_c in TRAINING_CMD:
            if t_c == '1\n':
                t_model = BILSTM
            elif t_c == '2\n':
                t_model = XGBOOST
            elif t_c == '3\n':
                t_model = LR
            elif t_c == '4\n':
                t_model = GRU
            elif t_c == '5\n':
                t_model = BIGRU
            elif t_c == '6\n':
                t_model = CNN
            elif t_c == '7\n':
                t_model = ATTENTION_BILSTM
            elif t_c == '8\n':
                t_model = CNN_TRANS_ENC
            else:
                t_model = BIGRU_CNN_TRANS_ENC

            print(f'Training {model}-{d_c}-{t_model}')
            run(m_c + d_c + ACT_CMD[1] + t_c)
            print(f'Finish Training {model}-{d_c}-{t_model}')
        print(f'Finish Training 9 models with dataset {d_c}, extraction model {model}')

print('NCKH Traning done, please take a look at results')
