import subprocess
import time

MODEL_CMD = ['1\n', '2\n', '3\n']
DATASET_CMD = ['1\n', '2\n', '3\n']
TRAINING_CMD = ['2\n1\n2\n1\n', '2\n1\n2\n2\n', '2\n1\n2\n3\n', '2\n1\n2\n4\n', '2\n1\n2\n5\n', '2\n1\n2\n6\n', '2\n1\n2\n7\n', '2\n1\n2\n8\n', '2\n1\n2\n9\n']

def run(command):
    process = subprocess.Popen(['python', '__init__.py'], stdin=subprocess.PIPE, text=True)
    process.communicate(command)
    
    print('Finish 1 automic process')
    time.sleep(5)

print('Do you wanna extract feature ?<y/n>')

key = input()

if key == 'y':
    print('Start Extracting Features')
    for m_c in MODEL_CMD:
        if m_c == '1\n':
            model = 'E2T-PhoBERT'
        elif m_c == '2\n':
            model = 'PhoBERT'
        else:
            model = 'VISOBERT'

        print(f'Start Extracting Features with extraction model {model}')
        for d_c in DATASET_CMD:
            print(f'Extracting Features with extraction model {model} and dataset {d_c}')
            run(m_c + d_c)
            print(f'Finish Extracting Features with extraction model {model} and dataset {d_c}')
        print(f'Finish Extracting Features with extraction model {model}')

    print('Finish Extracting Features')

for m_c in MODEL_CMD:
    if m_c == '1\n':
        model = 'E2T-PhoBERT'
    elif m_c == '2\n':
        model = 'PhoBERT'
    else:
        model = 'VISOBERT'

    for d_c in DATASET_CMD:
        print(f'Start Training 9 models with dataset {d_c}, extraction model {model}')
        for t_c in TRAINING_CMD:
            print(f'Training {model}-{d_c}-{t_c}')
            run(m_c + d_c + t_c)
            print(f'Finish Training {model}-{d_c}-{t_c}')
        print(f'Finish Training 9 models with dataset {d_c}, extraction model {model}')
