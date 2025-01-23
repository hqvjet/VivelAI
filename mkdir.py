import os
from constant import *

PRE = 'res/models'
FOLDERS = [AIVIVN, UIT_VIHSD, UIT_VSFC]

if not os.path.exists(PRE + '/' + E2T_PHOBERT):
    os.makedirs(PRE + '/' + E2T_PHOBERT)
    print('mkdir ' + PRE + '/' + E2T_PHOBERT)
else:
    print('mkdir ' + PRE + '/' + E2T_PHOBERT + ' already exists')

for folder in FOLDERS:
    if not os.path.exists(PRE + '/' + E2T_PHOBERT + '/' + folder):
        os.makedirs(PRE + '/' + E2T_PHOBERT + '/' + folder)
        print('mkdir ' + PRE + '/' + E2T_PHOBERT + '/' + folder)
    else:
        print('mkdir ' + PRE + '/' + E2T_PHOBERT + '/' + folder + ' already exists')

if not os.path.exists(PRE + '/' + E2V_PHOBERT):
    os.makedirs(PRE + '/' + E2V_PHOBERT)
    print('mkdir ' + PRE + '/' + E2V_PHOBERT)
else:
    print('mkdir ' + PRE + '/' + E2V_PHOBERT + ' already exists')

for folder in FOLDERS:
    if not os.path.exists(PRE + '/' + E2V_PHOBERT + '/' + folder):
        os.makedirs(PRE + '/' + E2V_PHOBERT + '/' + folder)
        print('mkdir ' + PRE + '/' + E2V_PHOBERT + '/' + folder)
    else:
        print('mkdir ' + PRE + '/' + E2V_PHOBERT + '/' + folder + ' already exists')

if not os.path.exists(PRE + '/' + PHOBERT):
    os.makedirs(PRE + '/' + PHOBERT)
    print('mkdir ' + PRE + '/' + PHOBERT)
else:
    print('mkdir ' + PRE + '/' + PHOBERT + ' already exists')

for folder in FOLDERS:
    if not os.path.exists(PRE + '/' + PHOBERT + '/' + folder):
        os.makedirs(PRE + '/' + PHOBERT + '/' + folder)
        print('mkdir ' + PRE + '/' + PHOBERT + '/' + folder)
    else:
        print('mkdir ' + PRE + '/' + PHOBERT + '/' + folder + ' already exists')

if not os.path.exists(PRE + '/' + VISOBERT):
    os.makedirs(PRE + '/' + VISOBERT)
    print('mkdir ' + PRE + '/' + VISOBERT)
else:
    print('mkdir ' + PRE + '/' + VISOBERT + ' already exists')

for folder in FOLDERS:
    if not os.path.exists(PRE + '/' + VISOBERT + '/' + folder):
        os.makedirs(PRE + '/' + VISOBERT + '/' + folder)
        print('mkdir ' + PRE + '/' + VISOBERT + '/' + folder)
    else:
        print('mkdir ' + PRE + '/' + VISOBERT + '/' + folder + ' already exists')
