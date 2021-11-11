'''
- Multi-fold RANDOM RAR/ROAR experiments to evaluate saliency methods.
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

from myscripts.createDirectories import create_Directories
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold

from roar_utils import rescale_two, rescale_input, compute_feature_ranking, random_feature_ranking, compute_feature_ranking_double_ended, weighted_feature_ranking_double_ended, compute_RAR_feature_ranking, random_perm_RAR_feature_ranking
   

Ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad', 4:'', 5:'smoothgrad', 6:'smoothgrad_sq', 7: 'vargrad'}

saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG'}

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

MODELS = {0: 'FPT', 1: 'UFPT', 2: 'NPT'}
Dataset = {0: 'FBIRN', 1: 'COBRE', 2: 'OASIS', 3: 'ABIDE'}
Directories = { 'COBRE': 'COBRESaliencies','FBIRN' : 'FBIRNSaliencies', 'ABIDE' : 'ABIDESaliencies','OASIS' : 'OASISSaliencies'}

FNCDict = {"FBIRN": LoadFBIRN, "COBRE": LoadCOBRE, "OASIS": LoadOASIS, "ABIDE": LoadABIDE}

Params = {'FBIRN':[311, 140, 121], 'COBRE': [157, 140, 121], 'OASIS': [372, 120, 101], 'ABIDE': [569, 140, 121]}

train_lim = { 'FBIRN': 250, 'COBRE': 100, 'OASIS': 300, 'ABIDE': 400}

Params_subj_distr = { 'FBIRN': [15, 25, 50, 75, 100], 'COBRE': [15, 25, 40], 'OASIS': [15, 25, 50, 75, 100, 120], 'ABIDE': [15, 25, 50, 75, 100, 150]}

test_lim_per_class = { 'FBIRN': 32, 'COBRE': 16, 'OASIS': 32, 'ABIDE': 50}


# Revised h_fixed Simiple Gain Selection

Params_best_gains = {1: {'FBIRN': {'NPT':[1.2, 1.3, 0.7, 1.1, 0.9], 'UFPT': [0.9, 0.7, 0.6, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.3, 1.3, 1.1, 0.9, 1.3, 0.6], 'UFPT': [0.4, 0.6, 0.5, 0.5, 0.3, 1.2]}, 'OASIS': {'NPT': [0.1, 1.2, 1.0, 1.0, 1.2, 0.7], 'UFPT': [0.7, 0.7, 1.3, 0.6, 1.1, 1.2]},'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [0.4, 1.1, 1.3]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}
                     
# ===============================================================================================================================


data= int(sys.argv[1])  # Choose FBIRN (0)/ COBRE (1) / OASIS (2) / ABIDE (3)
test_ID = int(sys.argv[2])



finalData, FNC, all_labels = FNCDict[Dataset[data]]()

print('Final Data Shape:', finalData.shape)
print('FNC shape:', FNC.shape)


HC_index, SZ_index = find_indices_of_each_class(all_labels)
print('Length of HC:', len(HC_index))
print('Length of SZ:', len(SZ_index))

# We will work here to change the split

test_starts = { 'FBIRN': [0, 32, 64, 96, 120], 'COBRE': [0, 16, 32, 48], 'OASIS': [0, 32, 64, 96, 120, 152], 'ABIDE': [0, 50, 100, 150, 200]}

test_indices = test_starts[Dataset[data]]
print(f'Test Start Indices: {test_indices}')


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


components = finalData.shape[1]
sample_y = 20
subjects = Params[Dataset[data]][0]
time_points = Params[Dataset[data]][1]
samples_per_subject = Params[Dataset[data]][2]
window_shift = 1

# dir = MODELS[mode] #'UFPT'   # NPT or UFPT

GAIN = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0} 


run_dir = "../wandb/Classic_ML_Models/"+Dataset[data]+"_FNC/New Results"
os.chdir(run_dir)   # Dynamically change the current directory during run time.


# sbpath = os.path.join('../../../', 'Sequence_Based_Models')
# basename = os.path.join(sbpath, Directories[Dataset[data]], dir, 'Saliency')

Trials = 10
np.random.seed(0)
trials_HC = [np.random.permutation(len(total_HC_index_tr)) for i in range(Trials)]  
trials_SZ = [np.random.permutation(len(total_SZ_index_tr)) for i in range(Trials)]  

models = ["MLP", "LinearSVM", "SVM", "LR"]

subjects_per_group = Params_subj_distr[Dataset[data]] 

# Best_gain = Params_best_gains[window_shift][Dataset[data]][MODELS[mode]]

name = "SVM" 

finalData = torch.from_numpy(finalData).float()
Fraction = [0, 0.05, 0.1, 0.2, 0.3] 


auc = np.zeros([len(Fraction), Trials])
accuracy = np.zeros([len(Fraction), Trials])

for restart in range(Trials):


    for f in range(len(Fraction)):

    # Remove data based on feature ranking and receive the updated data 

        Data = np.zeros(finalData.shape)
        for t in range(finalData.shape[0]):  
            Data[t] = random_perm_RAR_feature_ranking(finalData[t], finalData[t], Fraction[f])

        MaskedData = Data #finalData #*Map #Map1 #finalData*Map

        print('Running raw map sFNC experiments')
        FNCMasked = calculateFNC(MaskedData)

        print('Masked sFNC:', FNCMasked.shape)

        X = FNCMasked
        Y = all_labels.numpy()


        test_index = torch.cat((HC_index_test, SZ_index_test))
        test_index = test_index.view(test_index.size(0))
        X_test = X[test_index,:]
        Y_test = Y[test_index.long()]

        print('Test Shape', X_test.shape)


        # These three statements are if we again want to train in small data setting

    #     samples = subjects_per_group[i]
    #     HC_random = trials_HC[restart][:samples]  
    #     SZ_random = trials_SZ[restart][:samples]


        # These two lines if we want to use all the train data in evaluation (i.e. not again with small data)

        HC_random = trials_HC[restart] 
        SZ_random = trials_SZ[restart]


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


        print('Train Shape', X_train.shape)


        if name == "SVM" or name == "LinearSVM":
            # For SVM Trainer
            trainer = SVMTrainer(Y_train_go, Y_test, device, name)


        elif name == "LR":
            # For Logistic Regression Trainer
            trainer = LRTrainer(Y_train_go, Y_test, device)

        else:
            # For MLP classifier
            trainer = FCNetwork(Y_train_go, Y_test, device)

        accuracy[f, restart], auc[f, restart] = trainer.train(X_train_go, X_test, 0, restart, run_dir)


        print(auc)
        print(accuracy)


print('------------------------------------------Average Result---------------------------------------')
print('_'*80)

print(name+' AUC:\n' , auc)
print(name+' ACC:\n' , accuracy)


auc = pd.DataFrame(auc)
acc = pd.DataFrame(accuracy)


# For evaluation with RAR

fprefix = Dataset[data]+'_test_id_'+str(test_ID)+'_'+name+'_RAR'+'_new_RANDOM_perm_zero_fillings_sFNC_eval_1'

auc.to_csv(fprefix+'_AUC.csv')
acc.to_csv(fprefix+'_ACC.csv')

print("Average AUC:\n", auc.mean(axis=1).to_string(index=False))
print("Average ACC:\n", acc.mean(axis=1).to_string(index=False))

print('File Saved Here:', fprefix+'_AUC.csv')


elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())



