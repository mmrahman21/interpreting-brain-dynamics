'''
This code considers correct number of subjects distribution for the small data sFNC experiments
while the version run_ML_classifier.py is wrong in terms of subject distribution, otherwise that version is OK
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

using sklearn on small data maps
'''


import sys
import time
from collections import deque
from itertools import chain

from polyssifier import poly, polyr
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
from myscripts.LoadRealData import LoadABIDE, LoadCOBRE, LoadFBIRN, LoadOASIS
from myscripts.stitchWindows import stitch_windows
import pandas as pd
from datetime import datetime
from ML_HP_Tuned import SVMTrainer, DTTrainer, NBTrainer, RFTrainer, LRTrainer, KNNClassifier, FCNetwork


# run = int(sys.argv[1])
start_time = time.time()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index

def ReLU(x):
    print("Greetings from ReLU")
    return x * (x >= 0)

def calculateFNC(data):
    
    FNC = np.zeros((data.shape[0], 1378))
    corrM = np.zeros((data.shape[0], components, components))
    for k in range(data.shape[0]):
        corrM[k, :, :] = np.corrcoef(data[k])
        M = corrM[k, :, :]
#         print("Count Before: ", np.count_nonzero(np.isnan(M)))
        M = np.nan_to_num(M)
        FNC[k, :] = M[np.triu_indices(53, k=1)]
#         print("Count After: ", np.count_nonzero(np.isnan(FNC[k, :])))
    return FNC 


Dataset = {0: 'FBIRN', 1: 'COBRE', 2: 'OASIS', 3: 'ABIDE'}
Directories = { 'COBRE': 'COBRESaliencies','FBIRN' : 'FBIRNSaliencies', 'ABIDE' : 'ABIDESaliencies','OASIS' : 'OASISSaliencies'}

FNCDict = {"FBIRN": LoadFBIRN, "COBRE": LoadCOBRE, "OASIS": LoadOASIS, "ABIDE": LoadABIDE}

Params = {'FBIRN':[311, 140, 121], 'COBRE': [157, 140, 121], 'OASIS': [372, 120, 101], 'ABIDE': [569, 140, 121]}
train_lim = { 'FBIRN': 250, 'COBRE': 100, 'OASIS': 300, 'ABIDE': 400}

Params_subj_distr = { 'FBIRN': [15, 25, 50, 75, 100], 'COBRE': [15, 25, 40], 'OASIS': [15, 25, 50, 75, 100, 120], 'ABIDE': [15, 25, 50, 75, 100, 150]}

test_lim_per_class = { 'FBIRN': 32, 'COBRE': 15, 'OASIS': 32, 'ABIDE': 50}


data= 0  # Choose FBIRN (0)/ COBRE (1) / OASIS (2) / ABIDE (3)


finalData, FNC, all_labels = FNCDict[Dataset[data]]()

print('Final Data Shape:', finalData.shape)
print('FNC shape:', FNC.shape)


HC_index, SZ_index = find_indices_of_each_class(all_labels)
print('Length of HC:', len(HC_index))
print('Length of SZ:', len(SZ_index))

total_HC_index_tr = HC_index[:len(HC_index) - test_lim_per_class[Dataset[data]]]
total_SZ_index_tr = SZ_index[:len(SZ_index) - test_lim_per_class[Dataset[data]]]

print('Length of training HC:', len(total_HC_index_tr))
print('Length of training SZ:', len(total_SZ_index_tr))

HC_index_test = HC_index[len(HC_index) - (test_lim_per_class[Dataset[data]]):]
SZ_index_test = SZ_index[len(SZ_index) - (test_lim_per_class[Dataset[data]]):]

components = finalData.shape[1]
sample_y = 20
subjects = Params[Dataset[data]][0]
time_points = Params[Dataset[data]][1]
samples_per_subject = Params[Dataset[data]][2]
window_shift = 1


dir = 'NPT'   # NPT or UFPT

GAIN = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0} 


run_dir = "../wandb/Classic_ML_Models/FBIRN_FNC"
os.chdir(run_dir)   # Dynamically change the current directory during run time.


sbpath = os.path.join('../../', 'Sequence_Based_Models')
basename = os.path.join(sbpath, Directories[Dataset[data]], dir, 'Saliency')

Trials = 10
np.random.seed(0)
trials_HC = [np.random.permutation(len(total_HC_index_tr)) for i in range(Trials)]  
trials_SZ = [np.random.permutation(len(total_SZ_index_tr)) for i in range(Trials)]  

models = ["MLP", "LinearSVM", "SVM", "LR"]

subjects_per_group = [15, 25, 50, 75, 100]
# Best_gain = [0.2, 0.2, 0.3, 0.5, 0.4]   # For UFPT
Best_gain = [1.6, 0.8, 0.3, 0.9, 0.5]  # For FBIRN NPT
g = 1.0

# gain_ID = run
# name = models[run]

name = "LR"

# g = GAIN[run]

auc = np.zeros([len(subjects_per_group), Trials])
accuracy = np.zeros([len(subjects_per_group), Trials])

    
for i in range(len(subjects_per_group)):

    prefix = Dataset[data]+'_corr_subj_'+str(subjects_per_group[i])+'_selective_gain_'+str(Best_gain[i])+'_NPT_'+'LSTM_milc'

#     prefix = Dataset[data]+'_gain_'+str(g)+'_no_train_NPT_'+'LSTM_milc'

    for restart in range(Trials):

        filename1 = os.path.join(basename, prefix+'_both_scores_based_'+str(restart)+'.npy')
#         filename2 = os.path.join(basename, prefix+'_non_pred_'+str(restart)+'.npy')
        scorefilename = os.path.join(basename, prefix+'_both_scores_'+str(restart)+'.npy')

        print('Loaded saliency from...', filename1)
        Map1 = np.load(filename1)
        print(Map1.shape)
        
#         print('Loaded saliency from...', filename2)
#         Map2 = np.load(filename2)
#         print(Map2.shape)

        scores = np.load(scorefilename)
        print('Loaded scores from...', scorefilename)
        print('Scores:', scores.shape)
        scores = np.around(scores, 2)
        print('Score samples:', scores[0:10, :])

        Map1 = stitch_windows(Map1, components, samples_per_subject, sample_y, time_points, ws=window_shift)
        print('Average Shape:', Map1.shape)
        
#         Map2 = stitch_windows(Map2, components, samples_per_subject, sample_y, time_points, ws=window_shift)
#         print('Average Shape:', Map2.shape)


        Map1 = ReLU(Map1)
        Map1 = Map1/np.max(Map1)
        
#         print(Map1.mean())
#         print(Map1.std())

#         Map2 = ReLU(Map2)
#         Map2 = Map2/np.max(Map2)

    
#         Map = (Map1+Map2)*0.5
#         Map = Map1+Map2

        MaskedData = finalData*Map1 #Map1 #finalData*Map

        print('Running raw map sFNC experiments')
        FNCMasked = calculateFNC(MaskedData)

        print('Masked sFNC:', FNCMasked.shape)

        X = FNCMasked
        Y = all_labels.numpy()

#         avlTrData = X[:len(X) - 64]    # Remaining data for training
#         avlTrLabels = Y[:len(Y) - 64]

#         X_test = X[len(X)-64:]
#         Y_test = Y[len(Y)-64:]

        test_index = torch.cat((HC_index_test, SZ_index_test))
        test_index = test_index.view(test_index.size(0))
        X_test = X[test_index,:]
        Y_test = Y[test_index.long()]
        
        print('Test Shape', X_test.shape)

#         samples = 2*subjects_per_group[i]


#         X_train = avlTrData[trials[restart]][: samples]  
#         Y_train = avlTrLabels[trials[restart]][: samples]  

        samples = subjects_per_group[i]
        
        HC_random = trials_HC[restart][:samples]  
        SZ_random = trials_SZ[restart][:samples]
       

        HC_index_tr = total_HC_index_tr[HC_random]
        SZ_index_tr = total_SZ_index_tr[SZ_random]
        
        
        tr_index = torch.cat((HC_index_tr, SZ_index_tr))
        tr_index = tr_index.view(tr_index.size(0))
        X_train = X[tr_index, :]
        Y_train = Y[tr_index.long()]
        
        print('Train Shape', X_train.shape)


        if name == "SVM" or name == "LinearSVM":
            # For SVM Trainer
            trainer = SVMTrainer(Y_train, Y_test, device, name)


        elif name == "LR":
            # For Logistic Regression Trainer
            trainer = LRTrainer(Y_train, Y_test, device)

        else:
            # For MLP classifier
            trainer = FCNetwork(Y_train, Y_test, device)

        accuracy[i, restart], auc[i, restart] = trainer.train(X_train, X_test, subjects_per_group[i], restart, run_dir)
        
        print(auc)
        print(accuracy)

    
# print('------------------------------------------Average Result---------------------------------------')
# print('_'*80)

# print(models[run]+' AUC:' , auc)
# print(models[run]+' ACC:' , accuracy)

# auc = pd.DataFrame(auc)
# acc = pd.DataFrame(accuracy)

# fprefix = Dataset[data]+'_'+models[run]+'_small_subj_'+'NPT_'+'LSTM_milc'
# auc.to_csv(fprefix+'_corr_subj_all_restarts_relu_map_masked_data_sFNC_classification_AUC_corr.csv')
# acc.to_csv(fprefix+'_corr_subj_all_restarts_relu_map_masked_data_sFNC_classification_ACC_corr.csv')

# print('File Saved Here:', fprefix+'_corr_subj_all_restarts_relu_map_masked_data_sFNC_classification_metric_corr.csv')
# elapsed_time = time.time() - start_time
# print('Total Time Elapsed:', elapsed_time)
# print(datetime.now())



