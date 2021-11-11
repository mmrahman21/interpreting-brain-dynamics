from myscripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2, \
    artificial_batching_trend, multi_way_data, my_multi_block_var_data, actual_spatial, three_class, my_uniform_top_down

from myscripts.LoadRealData import LoadABIDE, LoadOASIS, LoadCOBRE, LoadFBIRN
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold

import numpy as np
import torch

data_info = { 0 : { 'dataset' : {0: 'oldtwocls', 1: 'newtwocls', 2: 'threecls', 3: 'multiwaytwocls', 4: 'uniformtwocls'}, 
                  'dataset_func' : {"oldtwocls": artificial_batching_patterned_space2, "newtwocls": actual_spatial, "threecls": 
                                   three_class, "multiwaytwocls": multi_way_data, "uniformtwocls": my_uniform_top_down},
                  'directories' : { 'oldtwocls': '','newtwocls' : 'newTwoClass', 'threecls' : 'threeClass', 
                                       'multiwaytwocls':'multiwayTwoClass', 'uniformtwocls': ''}
                 }, 1 : { 'dataset' : {0: 'FBIRN', 1: 'COBRE', 2: 'OASIS', 3: 'ABIDE'}, 'dataset_func' : {"FBIRN": LoadFBIRN, "COBRE": LoadCOBRE, "OASIS": LoadOASIS, "ABIDE": LoadABIDE}, 'directories' : { 'COBRE': 'COBRESaliencies','FBIRN' : 'FBIRNSaliencies', 'ABIDE' : 'ABIDESaliencies','OASIS' : 'OASISSaliencies'}}
             }


def format_data(input_data, window_shift = 1):
    
    '''Takes raw data and based on window shift, returns the reformatted data'''
    
    no_good_comp = input_data.shape[1]
    sample_y = 20
    subjects = input_data.shape[0]
    tc = input_data.shape[2]
    samples_per_subject = (tc - 20) // window_shift + 1

    formatted_data = np.zeros((subjects, samples_per_subject, no_good_comp, sample_y))

    for i in range(subjects):
        for j in range(samples_per_subject):
            formatted_data[i, j, :, :] = input_data[i, :, (j * window_shift):(j * window_shift) + sample_y]
            
    return formatted_data

def split_synthetic_data(X, Y, train_size, test_size, sal_start_index, device):
    
    '''Takes a dataset X, splits into train (from start), test (after train) and sal (sal data start index).'''
    X_train = X[:train_size] # X[kf[fold][0]]  
    Y_train = Y[:train_size] # Y[kf[fold][0]]  

    X_train = torch.from_numpy(X_train).float().to(device)
    Y_train = torch.from_numpy(Y_train).long().to(device)

    X_test = X[train_size:train_size+test_size] # X[kf[fold][1]] 
    Y_test = Y[train_size:train_size+test_size] # Y[kf[fold][1]] 

    X_sal = X[sal_start_index:]
    Y_sal = Y[sal_start_index:]

    X_test = torch.from_numpy(X_test).float().to(device)
    Y_test = torch.from_numpy(Y_test).long().to(device)

    X_sal = torch.from_numpy(X_sal).float().to(device)
    Y_sal = torch.from_numpy(Y_sal).long().to(device)

    return X_train, Y_train, X_test, Y_test, X_sal, Y_sal

    
# Params = {'FBIRN':[311, 140, 121], 'COBRE': [157, 140, 121], 'OASIS': [372, 120, 101], 'ABIDE': [569, 140, 121]}
# train_lim = { 'FBIRN': 250, 'COBRE': 100, 'OASIS': 300, 'ABIDE': 400}



