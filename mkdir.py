import os
from constant import *

PRE = 'res/models'

if not os.path.exists(PRE + '/' + E2T_PHOBERT):
    os.makedirs(PRE + '/' + E2T_PHOBERT)
    print('mkdir ' + PRE + '/' + E2T_PHOBERT)
else:
    print('mkdir ' + PRE + '/' + E2T_PHOBERT + ' already exists')

if not os.path.exists(PRE + '/' + E2V_PHOBERT):
    os.makedirs(PRE + '/' + E2V_PHOBERT)
    print('mkdir ' + PRE + '/' + E2V_PHOBERT)
else:
    print('mkdir ' + PRE + '/' + E2V_PHOBERT + ' already exists')

if not os.path.exists(PRE + '/' + PHOBERT):
    os.makedirs(PRE + '/' + PHOBERT)
    print('mkdir ' + PRE + '/' + PHOBERT)
else:
    print('mkdir ' + PRE + '/' + PHOBERT + ' already exists')

if not os.path.exists(PRE + '/' + VISOBERT):
    os.makedirs(PRE + '/' + VISOBERT)
    print('mkdir ' + PRE + '/' + VISOBERT)
else:
    print('mkdir ' + PRE + '/' + VISOBERT + ' already exists')
