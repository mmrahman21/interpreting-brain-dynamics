'''
This code is intended to run Standard Machine Learning algorithms directly on raw ICA timecourses in small data regime.
'''

'''
FNC/ICA+SVM
FNC/ICA+Decision Tree
FNC/ICA+Random Forests
FNC/ICA+Naive Bayes
FNC/ICA+Logistic Regression
FNC/ICA+KNN
FNC/ICA+Multilayer Perceptron

Datasets:
FBIRN
COBRE
OASIS
ABIDE

using sklearn
'''


import sys
import time
from collections import deque
from itertools import chain

# from polyssifier import poly, polyr
import numpy as np
import torch

import datetime
import os

from myscripts.createDirectories import  create_Directories
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import random
import h5py
import time
import pickle as p
from myscripts.LoadRealData import LoadABIDE, LoadCOBRE, LoadFBIRN, LoadOASIS, LoadAddiction
from myscripts.stitchWindows import stitch_windows
import pandas as pd
from datetime import datetime
from ML_HP_Tuned import SVMTrainer, DTTrainer, NBTrainer, RFTrainer, LRTrainer, KNNClassifier, FCNetwork
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index



Dataset = {0: 'FBIRN', 1: 'COBRE', 2: 'OASIS', 3: 'ABIDE', 4 : 'ADDICTION'}
Directories = { 'COBRE': 'COBRE_FNC','FBIRN' : 'FBIRN_FNC', 'ABIDE' : 'ABIDE_FNC','OASIS' : 'OASIS_FNC', 'ADDICTION' : 'ADDICTION_FNC'}

FNCDict = {"FBIRN": LoadFBIRN, "COBRE": LoadCOBRE, "OASIS": LoadOASIS, "ABIDE": LoadABIDE, "ADDICTION": LoadAddiction}

Params = {'FBIRN':[311, 140, 121], 'COBRE': [157, 140, 121], 'OASIS': [372, 120, 101], 'ABIDE': [569, 140, 121], 'ADDICTION': [706, 140, 121]}

train_lim = { 'FBIRN': 250, 'COBRE': 100, 'OASIS': 300, 'ABIDE': 400, 'ADDICTION': 500}

Params_subj_distr = { 'FBIRN': [15, 25, 50, 75, 100], 'COBRE': [15, 25, 40], 'OASIS': [15, 25, 50, 75, 100, 120], 'ABIDE': [15, 25, 50, 75, 100, 150], 'ADDICTION': [15, 25, 50, 75, 100, 200]}

test_lim_per_class = { 'FBIRN': 32, 'COBRE': 16, 'OASIS': 32, 'ABIDE': 50, 'ADDICTION': 50}
val_lim_per_class = { 'FBIRN': 16, 'COBRE': 16, 'OASIS': 32, 'ABIDE': 32, 'ADDICTION': 50}


                     
data= int(sys.argv[1])  # Choose FBIRN (0)/ COBRE (1) / OASIS (2) / ABIDE (3) / ADDICTION (4)


finalData, FNC, all_labels = FNCDict[Dataset[data]]()
print('Final Data Shape:', finalData.shape)
print('FNC Data Shape:', FNC.shape)

flattened_data = finalData.reshape((len(all_labels), -1))
print('Flattened Shape:', flattened_data.shape)


subjects = Params[Dataset[data]][0]
tc = Params[Dataset[data]][1]


HC_index, SZ_index = find_indices_of_each_class(all_labels)
print('Length of HC:', len(HC_index))
print('Length of SZ:', len(SZ_index))

# We will work here to change the split

test_starts = { 'FBIRN': [0, 32, 64, 96, 120], 'COBRE': [0, 16, 32, 48], 'OASIS': [0, 32, 64, 96, 120, 152], 'ABIDE': [0, 50, 100, 150, 200], 'ADDICTION': [0, 50, 100, 150, 200]}

test_indices = test_starts[Dataset[data]]
print(f'Test Start Indices: {test_indices}')

test_ID = int(sys.argv[2])

test_start_index = test_indices[test_ID]
test_end_index = test_start_index + test_lim_per_class[Dataset[data]]
total_HC_index_tr = torch.cat(
    [HC_index[:test_start_index], HC_index[test_end_index:]]
)
total_SZ_index_tr = torch.cat(
    [SZ_index[:test_start_index], SZ_index[test_end_index:]]
)

HC_index_test = HC_index[test_start_index:test_end_index]
SZ_index_test = SZ_index[test_start_index:test_end_index]

# total_HC_index_tr = HC_index[:len(HC_index) - test_lim_per_class[Dataset[data]]]
# total_SZ_index_tr = SZ_index[:len(SZ_index) - test_lim_per_class[Dataset[data]]]

print('Length of training HC:', len(total_HC_index_tr))
print('Length of training SZ:', len(total_SZ_index_tr))

# HC_index_test = HC_index[len(HC_index) - (test_lim_per_class[Dataset[data]]):]
# SZ_index_test = SZ_index[len(SZ_index) - (test_lim_per_class[Dataset[data]]):]

        
X = FNC # flattened_data
Y = all_labels.numpy()

print('Data shape ended up with:', X.shape)

print("with LSTM enc + unidirectional lstm MILC + anchor point + sequence first...")


Trials = 10
np.random.seed(0)
trials_HC = [np.random.permutation(len(total_HC_index_tr)) for i in range(Trials)]  
trials_SZ = [np.random.permutation(len(total_SZ_index_tr)) for i in range(Trials)]  


subjects_per_group = Params_subj_distr[Dataset[data]] 

print('SPC Info:', subjects_per_group)

test_index = torch.cat((HC_index_test, SZ_index_test))
test_index = test_index.view(test_index.size(0))
X_test = X[test_index, :]
Y_test = Y[test_index.long()]
                         

print('Test Shape:', X_test.shape)


accMat = np.zeros([len(subjects_per_group), Trials])
aucMat = np.zeros([len(subjects_per_group), Trials])


wdb = 'wandb'
wpath = os.path.join('../', wdb)
sbpath = os.path.join(wpath, 'Classic_ML_Models')

model_path = os.path.join(sbpath, Directories[Dataset[data]])
# os.chdir(model_path)   # Dynamically change the current directory during run time.

models = ["MLP", "LinearSVM", "SVM", "LR", "RF"]

for name in models:
    start_time = time.time()

    for i in range(5, len(subjects_per_group), 1):
        for restart in range(Trials):

            samples = subjects_per_group[i]

            HC_random = trials_HC[restart][:samples]  
            SZ_random = trials_SZ[restart][:samples]


            HC_index_tr = total_HC_index_tr[HC_random]
            SZ_index_tr = total_SZ_index_tr[SZ_random]


            tr_index = torch.cat((HC_index_tr, SZ_index_tr))
            tr_index = tr_index.view(tr_index.size(0))
            X_train = X[tr_index, :]
            Y_train = Y[tr_index.long()]


            np.random.seed(0)
            randomize = np.random.permutation(X_train.shape[0])

            X_train_go = X_train[randomize]
            Y_train_go = Y_train[randomize]

            print(f'\n\n\nTest Split Starts: {test_ID}\nRunning Model: {name}')
            print(f'Model Started: {restart}\nSPC: {samples}\nDataset: {Dataset[data]}')
            print('Train Data Shape:', X_train.shape)



            if name == "SVM" or name == "LinearSVM":
                # For SVM Trainer
                trainer = SVMTrainer(Y_train_go, Y_test, device, name)

            elif name == "LR":
                # For Logistic Regression Trainer
                trainer = LRTrainer(Y_train_go, Y_test, device)

            elif name == "RF":
                trainer = RFTrainer(Y_train_go, Y_test, device)

            else:
                # For MLP classifier
                trainer = FCNetwork(Y_train_go, Y_test, device)

            accMat[i, restart], aucMat[i, restart] = trainer.train(X_train_go, X_test, subjects_per_group[i], restart, model_path)

            middle_time = time.time() - start_time
            print('Total Time for Training:', middle_time)


#             prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_ICA_tc_test_id_{test_ID}_model_{name}'



    basename2 = os.path.join(sbpath, Directories[Dataset[data]])

#     prefix = f'{Dataset[data]}_all_spc_ICA_tc_test_id_{test_ID}_model_{name}'
    
    prefix = f'{Dataset[data]}_all_spc_sFNC_test_id_{test_ID}_model_{name}'


    accDataFrame = pd.DataFrame(accMat)
    accfname = os.path.join(basename2, prefix +'_ACC.csv')
    accDataFrame.to_csv(accfname)
    print('Result Saved Here:', accfname)

    aucDataFrame = pd.DataFrame(aucMat)
    aucfname = os.path.join(basename2, prefix +'_AUC.csv')
    aucDataFrame.to_csv(aucfname)

    print("AUC:", aucMat)
    print("ACC:", accMat)

    elapsed_time = time.time() - start_time
    print('Total Time Elapsed:', elapsed_time)
    print(datetime.now())






