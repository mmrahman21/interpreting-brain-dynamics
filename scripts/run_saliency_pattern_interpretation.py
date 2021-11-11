'''
THIS IS NOT FOR SALIENCY EVALUATION, ONLY FOR sFNC VISUALIZATIOON AND INTERPRETATION
Saliency Interpretation
Find the 5%-10%
Plot sFNC
Observe Patterns
'''

import sys
import time


import numpy as np
import torch
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import seaborn as sb

import datetime
import os


import pandas as pd

import random
import h5py
import time

from myscripts.LoadRealData import LoadABIDE, LoadCOBRE, LoadFBIRN, LoadOASIS
from myscripts.stitchWindows import stitch_windows
from datetime import datetime

from roar_utils import rescale_two, rescale_input, compute_feature_ranking, random_feature_ranking, compute_feature_ranking_double_ended, weighted_feature_ranking_double_ended, compute_RAR_feature_ranking, random_perm_RAR_feature_ranking

from myscripts.interpretation_utilities import plot_dfnc, plot_average_dfnc, plot_pattern, plot_binary_mask, plot_avg_pattern, do_t_test


saliency_id = int(sys.argv[1])   

ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad', 4:'', 5:'smoothgrad', 6:'smoothgrad_sq', 7: 'vargrad'}

saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime'}


start_time = time.time()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index

def ReLU(x):
    print("Greetings from ReLU")
    return x * (x >= 0)

def dFNC(data):
       
    M = np.corrcoef(data) 
    M = np.nan_to_num(M)
      
    return M 

def calculateFNC(data):
    
    FNC = np.zeros((data.shape[0], 1378))
    corrM = np.zeros((data.shape[0], components, components))
    for k in range(data.shape[0]):
        corrM[k, :, :] = np.corrcoef(data[k])
        M = corrM[k, :, :]
#         print("Count Before: ", np.count_nonzero(np.isnan(M)))
        M = np.nan_to_num(M)
        FNC[k, :] = M[np.triu_indices(53, k=1)]
        corrM[k, :, :]= M
#         print("Count After: ", np.count_nonzero(np.isnan(FNC[k, :])))
    return FNC, corrM 


MODELS = {0: 'FPT', 1: 'UFPT', 2: 'NPT'}

Dataset = {0: 'FBIRN', 1: 'COBRE', 2: 'OASIS', 3: 'ABIDE'}
Directories = { 'COBRE': 'COBRESaliencies','FBIRN' : 'FBIRNSaliencies', 'ABIDE' : 'ABIDESaliencies','OASIS' : 'OASISSaliencies'}

FNCDict = {"FBIRN": LoadFBIRN, "COBRE": LoadCOBRE, "OASIS": LoadOASIS, "ABIDE": LoadABIDE}

Params = {'FBIRN':[311, 140, 121], 'COBRE': [157, 140, 121], 'OASIS': [372, 120, 101], 'ABIDE': [569, 140, 121]}
train_lim = { 'FBIRN': 250, 'COBRE': 100, 'OASIS': 300, 'ABIDE': 400}

Params_subj_distr = { 'FBIRN': [15, 25, 50, 75, 100], 'COBRE': [15, 25, 40], 'OASIS': [15, 25, 50, 75, 100, 120], 'ABIDE': [15, 25, 50, 75, 100, 150]}

test_lim_per_class = { 'FBIRN': 32, 'COBRE': 15, 'OASIS': 32, 'ABIDE': 50}

# These gains were based on cross-validation based search

Params_best_gains = {1: {'FBIRN': {'NPT':[1.0, 0.9, 0.9, 1.0, 1.0], 'UFPT': [1.0, 0.8, 0.7, 0.5, 0.4]}, 'OASIS': {'NPT': [0.9, 0.9, 1.0, 1.1, 1.1, 1.0], 'UFPT': [0.8, 0.8, 0.4, 0.3, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.0, 0.9, 0.9, 1.0, 0.7, 0.6], 'UFPT': [0.4, 0.8, 0.3, 0.2, 0.2, 0.1]}, 'COBRE': {'NPT': [1.1, 0.7, 0.8], 'UFPT': [1.6, 1.3, 1.3]}}, 
                     10: {'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [1.2, 1.2, 1.4]}},
                     20: {'COBRE': {'NPT': [1.5, 0.8, 0.8], 'UFPT': [1.2, 1.3, 1.3]}}  
                    }

# Revised h_fixed Simiple Gain Selection

# Params_best_gains = {1: {'FBIRN': {'NPT':[1.2, 1.3, 0.7, 1.1, 0.9], 'UFPT': [0.9, 0.7, 0.6, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.3, 1.3, 1.1, 0.9, 1.3, 0.6], 'UFPT': [0.4, 0.6, 0.5, 0.5, 0.3, 1.2]}, 'OASIS': {'NPT': [0.1, 1.2, 1.0, 1.0, 1.2, 0.7], 'UFPT': [0.7, 0.7, 1.3, 0.6, 1.1, 1.2]},'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [0.4, 1.1, 1.3]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}

mode = 1  # Choose FPT (0) / UFPT (1) / NPT (2)
data= 3  # Choose FBIRN (0)/ COBRE (1) / OASIS (2) / ABIDE (3)


finalData, FNC, all_labels = FNCDict[Dataset[data]]()

print('Final Data Shape:', finalData.shape)
print('FNC shape:', FNC.shape)

HC_index, SZ_index = find_indices_of_each_class(all_labels)

print('Length of HC:', len(HC_index))
print('Length of SZ:', len(SZ_index))

print('HC_index:', HC_index.shape)
print('SZ_index:', SZ_index.shape)

HC_index = np.squeeze(HC_index, axis = 1)
SZ_index = np.squeeze(SZ_index, axis = 1)
print('HC_index:', HC_index.shape)


components = finalData.shape[1]
sample_y = 20
subjects = Params[Dataset[data]][0]
time_points = Params[Dataset[data]][1]
samples_per_subject = Params[Dataset[data]][2]
window_shift = 1


dir = MODELS[mode]   # NPT or UFPT

GAIN = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0} 


run_dir = "../wandb/Classic_ML_Models/"+Dataset[data]+"_FNC"
os.chdir(run_dir)   # Dynamically change the current directory during run time.


sbpath = os.path.join('../../', 'Sequence_Based_Models')
basename = os.path.join(sbpath, Directories[Dataset[data]], dir, 'Saliency')

Trials = 10
np.random.seed(0)

models = ["MLP", "LinearSVM", "SVM", "LR"]

subjects_per_group = Params_subj_distr[Dataset[data]] 

Best_gain = Params_best_gains[window_shift][Dataset[data]][MODELS[mode]]
print('Best Gain Chosen:', Best_gain)

# Best_gain = [0.2, 0.2, 0.3, 0.5, 0.4]   # For UFPT
# Best_gain = [1.6, 0.8, 0.3, 0.9, 0.5]  # For FBIRN NPT


# g = 0.8


finalData = torch.from_numpy(finalData).float()

Fraction = [0.05, 0.1, 0.2, 0.3] 



fold = 0 #int(sys.argv[1])

# for i in range(len(subjects_per_group)):

i = 5
g = Best_gain[i]

prefix = Dataset[data]+'_spc_'+str(subjects_per_group[i])+'_cross_val_gain_'+str(g)+'_'+dir+'_LSTM_milc'

# prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_simple_gain_{g}_window_shift_{window_shift}_{dir}_arch_chk_h_fixed_test_id_{test_ID}_LSTM_milc'

# prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_cross_val_gain_{g}_window_shift_{window_shift}_{dir}_LSTM_milc'
# prefix = Dataset[data]+'_fold_'+str(fold)+'_gain_'+str(g)+'_NPT_LSTM_milc_prediction'

#     base_estimator = "IG"

dissimilar_components = np.zeros((10, 53, 2))

for restart in range(Trials):  # Trials

    predictions_file = os.path.join(basename, prefix+'_labels_save_'+str(restart)+'.npy')

    predictions = np.load(predictions_file)
#     print('Predictions Shape:', predictions.shape)

    filename1 = os.path.join(basename, prefix+'_prediction_'+saliency_options[saliency_id]+'_'+str(restart)+'.npy')

    print('Loaded saliency from...', filename1)
    Map1 = np.load(filename1)
#     print(Map1.shape)


    Map1 = stitch_windows(Map1, components, samples_per_subject, sample_y, time_points, ws=window_shift)
#     print('Average Shape:', Map1.shape)


#     It ensures that while creating feature_ranking mask, the zero values will be treated separately 

    epsilon = 0.000000001
    Map1[Map1 == 0] = Map1[Map1 == 0] + epsilon
    
    sal_map = Map1 / np.max(Map1)
    sal_map = torch.from_numpy(sal_map).float()


    Map1 = torch.from_numpy(Map1).float()

    for f in range(len(Fraction)):

    # Remove data based on feature ranking and receive the updated data 

        Data = np.zeros(finalData.shape)

        data_mask = np.zeros(finalData.shape)

        component_counter = np.zeros((finalData.shape[0], 53))

        # For revised data for saliency interpretation

        dCorrM1 = np.zeros((finalData.shape[0], 53, 53))  # Dynamic correlation matrices for all subjects. 
        dCorrM2 = np.zeros((finalData.shape[0], 53, 53))  # Dynamic correlation matrices for all subjects. 

        for t in range(Map1.shape[0]):
            Data[t] = compute_RAR_feature_ranking(finalData[t], Map1[t], Fraction[f])
            result = np.where(Data[t] != 0.0)
#             print(result)

#             print(Data[t])
            data_mask[t][Data[t] !=0] = 1 
#             print(data_mask[t])

#             print(np.sum(Data[t], axis=1).shape)
            component_counter[t, :] = np.sum(data_mask[t], axis=1)
#             print(component_counter[t])

      
        Labels = all_labels.numpy()

        D = np.asarray(range(490, 500, 1))
        print(D)
        L = Labels[D[:]]
        print('Labels:', L)

        x_HC = data_mask[HC_index, :]
        x_PT = data_mask[SZ_index, :]

        print('Healthy Shape:', x_HC.shape)
        print('Patient Shape:', x_PT.shape)


        X_HC_t_test_input = component_counter[HC_index, :]
        X_PT_t_test_input = component_counter[SZ_index, :]
        
        print('Healthy t-test shape:', X_HC_t_test_input.shape)
        print('Healthy t-test shape:', X_HC_t_test_input.shape)

#         dissimilar_components[restart, :, f] = do_t_test(X_HC_t_test_input, X_PT_t_test_input)



        path = os.path.join(sbpath, Directories[Dataset[data]], dir, 'Maps')

        # Plot group-wise pattern using sum of binary masks

#         pattern_filename = f"{prefix}_{saliency_options[saliency_id]}_MOD_{restart}_frac_{Fraction[f]}_percent_group_wise_max_pattern.png"
#         group_wise_pattern_path = os.path.join(path, pattern_filename)
#         print(group_wise_pattern_path)
#         plot_avg_pattern(x_HC, x_PT, group_wise_pattern_path)

        # Plot individual pattern based on binary masks

        pattern_filename = f"{prefix}_{saliency_options[saliency_id]}_MOD_{restart}_frac_{Fraction[f]}_percent_map_for_draft.svg"
        pattern_path = os.path.join(path, pattern_filename)
        print(pattern_path)

        data_mask = torch.from_numpy(data_mask).float()

        masked_sal = sal_map*data_mask # finalData*data_mask   
        finalData = finalData/torch.max(finalData)
        plot_pattern(finalData, masked_sal, sal_map, Labels, D, L, predictions, pattern_path)
#         plot_binary_mask(data_mask, Labels, D, L, predictions, pattern_path)

# print(np.sum(dissimilar_components, axis=0))       
    
elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())


          
#             print("Unique components shape:", np.unique(result[0]).shape)
#             print("Unique time steps shape:", np.unique(result[1]).shape)
#             times = np.unique(result[1])  # Working on important time steps
#             print('Time Steps:', result[1])
#             print("Unique Time Steps:", times)
            
#             temp1 = finalData[t][:, times] # Only consider important (5%) time steps
#             temp2 = Data[t][:, times]      # RAR 5%
#             dCorrM1[t, :, :] = dFNC(temp1)    # Taking full 5% timesteps 
#             dCorrM2[t, :, :] = dFNC(temp2)    # Taking only 5% entries
    
        
#         MaskedData = Data # finalData  
#         print('Calculating sFNC')
#         FNCMasked, M = calculateFNC(MaskedData)

#  X = FNCMasked

       
#         print('Masked sFNC:', FNCMasked.shape)

#         FinalMatrix = dCorrM2 #dCorrM2 #dCorrM2, M
        
#         print('Cov Matrix Shape:', FinalMatrix.shape)
# X_HC = FinalMatrix[HC_index, :, :]
# X_PT = FinalMatrix[SZ_index, :, :]

        # Plot group-based average dFNC based on RAR
#         file_name1 = f"{prefix}_{saliency_options[saliency_id]}_MOD_{restart}_frac_{Fraction[f]}_percent_avg_dFNC.png"
#         path1 = os.path.join(path, file_name1)
#         print(path1)
#         plot_average_dfnc(X_HC, X_PT, path1)
        
        # Plot individual dFNC based on RAR
#         file_name2 = f"{prefix}_{saliency_options[saliency_id]}_MOD_{restart}_frac_{Fraction[f]}_percent_dFNC.png"
#         path2 = os.path.join(path, file_name2)
#         plot_dfnc(FinalMatrix, Labels, D, L, predictions, path2)
 