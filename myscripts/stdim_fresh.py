'''
This code is used to run downstream experiments in low data regime using with/without stdim pre-trained encoder
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

from source.utils import get_argparser
from source.encoders_ICA import NatureCNN, ImpalaCNN, NatureOneCNN, LinearEncoder

from myscripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2, \
    artificial_batching_trend, multi_way_data, my_multi_block_var_data, actual_spatial, three_class

from myscripts.LoadRealData import LoadABIDE, LoadOASIS, LoadCOBRE, LoadFBIRN
from source.ATTLSTM import subjLSTM, onlyLSTM
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from myscripts.milc_utilities_handler import load_pretrain_model, get_captum_saliency, save_predictions, save_reload_model,save_acc_auc

from datetime import datetime
import pandas as pd

import sys
import os
import time
import gc

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    ShapleyValues,
    ShapleyValueSampling,
    Lime,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion, 
    Saliency,
    GuidedBackprop,
)



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
args = Namespace(end_with_relu=False, feature_size=256, no_downsample=True, script_ID = 0)


GAIN = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0}  


# ID = int(sys.argv[1])
# saliency_id = int(sys.argv[1])   

ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad', 4:'', 5:'smoothgrad', 6:'smoothgrad_sq', 7: 'vargrad'}

saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime'}


start_time = time.time()

device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

print(device)

# np.random.seed(run)

def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


class LSTM(torch.nn.Module):

    def __init__(self, encoder, feature_size, hidden_nodes, output_size, gain):
        super(LSTM, self).__init__()
       
        self.hidden_dim = hidden_nodes
        
        self.lstm = nn.LSTM(feature_size, hidden_nodes // 2, batch_first=True, bidirectional = True)
        

        self.encoder = encoder
        
#         self.attn = nn.Sequential(
#             nn.Linear(2*self.hidden_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 200),
            nn.ReLU(),
            nn.Linear(200, output_size), 
            nn.Sigmoid()
        )
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing All components')

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
                
#         for name, param in self.attn.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_normal_(param, gain=self.gain)
                
        for name, param in self.decoder.named_parameters():
            print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
                
    def add_regularization(self, loss):
        
#         print('Adding regularizer...')
        reg = 1e-3
        l2_reg = torch.tensor(0.).to(device)
        
        for name, param in self.encoder.named_parameters():
            if 'bias' not in name:
                l2_reg += torch.norm(param)
                
        for name, param in self.lstm.named_parameters():
            if 'bias' not in name:
                l2_reg += torch.norm(param)
                
        for name, param in self.decoder.named_parameters():
            if 'bias' not in name:
                l2_reg += torch.norm(param)
                
               
        return loss + reg*l2_reg
                
    def init_hidden(self, batch_size, device):
        
        h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2, device=device))
        c0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2, device=device))
        
        return (h0, c0)

    def forward(self, x):
        
        sx = []
        for episode in x:
            mean = episode.mean()
            sd = episode.std()
            episode = (episode - mean) / sd
            sx.append(episode)

        x = torch.stack(sx)
        
        b_size = x.size(0)
        s_size = x.size(1)
        x = x.view(-1, x.shape[2], 20)
        
        out = self.encoder(x)
        
        out = out.view(b_size, s_size, -1)
        
        self.hidden = self.init_hidden(b_size, device)
        lstm_out, self.hidden = self.lstm(out, self.hidden)
        
        # Attention added on lstm top
        
#         lstm_out = self.get_attention(lstm_out)
#         lstm_out = lstm_out.view(b_size, -1)
        
        outputs = [self.decoder(torch.cat((hid[0, self.hidden_dim // 2:],
                                           hid[-1, :self.hidden_dim // 2]), 0)) for hid in lstm_out]
        
        outputs = torch.stack(outputs)
        
        
        return outputs # lstm_out # outputs

    def get_attention(self, outputs):
        
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.hidden_dim)

        weights = self.attn(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()

        # Pass the weighted output to decoder
        logits = self.decoder(attn_applied)
        return logits
    
    def get_attention_enc(self, outputs):
        
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.enc_out)

        weights = self.attnenc(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()

        return attn_applied


class DataWithLabels(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.Y[i]


def get_data_loader(X, Y, batch_size):
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle = True)

    return dataLoader


def update_lr(learning_rate0, epoch_num, decay_rate):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer
    decay_rate -- Decay rate. Scalar

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    
    learning_rate = (1./(1 + decay_rate * epoch_num)) * learning_rate0
    
    return learning_rate


def schedule_lr_decay(optimizer, learning_rate0, epoch_num, decay_rate=1, time_interval=1000):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    
    if epoch_num % time_interval == 0:
        learning_rate = (1./(1 + decay_rate * (epoch_num / time_interval)))*learning_rate0
    else:
        learning_rate = optimizer.param_groups[0]["lr"]
    
    return learning_rate


def train_model(model, loader_train, loader_train_check, loader_test, epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):
        
        learn_rate = schedule_lr_decay(optimizer, learning_rate, epoch, decay_rate=1, time_interval = 100)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learn_rate
            
        cur_lr = optimizer.param_groups[0]["lr"] 
       

        for i, data in enumerate(loader_train):
            x, y = data
            # x = x.permute(1, 0, 2)
            optimizer.zero_grad()
            outputs= model(x)
           
            l = loss(outputs, y)
            
            # Add regularizer
            l = model.add_regularization(l)
            
            _, preds = torch.max(outputs.data, 1)
            accuracy = (preds == y).sum().item()
            l.backward()

            optimizer.step()
        x_test, y_test = next(iter(loader_test))
        # x_test = x_test.permute(1, 0, 2)
        outputs = model(x_test)
        _, preds = torch.max(outputs.data, 1)
        accuracy = (preds == y_test).sum().item()
        
        sig = F.softmax(outputs, dim=1).to(device)
        y_scores = sig.detach()[:, 1]
        roc = roc_auc_score(y_test.to('cpu'), y_scores.to('cpu'))
        
        x_train, y_train = next(iter(loader_train_check))
        train_outputs = model(x_train)
        _, train_preds = torch.max(train_outputs.data, 1)
        train_accuracy = (train_preds == y_train).sum().item()
        train_accuracy /= y_train.size(0)
        
        train_sig = F.softmax(train_outputs, dim=1).to(device)
        y_train_scores = train_sig.detach()[:, 1]
        train_roc = roc_auc_score(y_train.to('cpu'), y_train_scores.to('cpu'))
        
        
        
        print("epoch: " + str(epoch) + ", loss: " + str(l.detach().item()) +", test acc: " + str(accuracy / y_test.size(0)) + ", roc: " + str(roc) +", train acc: " + str(train_accuracy) +" , train roc: " + str(train_roc)+", learn rate: "+ str(cur_lr))
        
#         test_loss = loss(outputs, y_test)
#         scheduler.step(test_loss)
        
        
    return optimizer, accuracy / y_test.size(0), roc


            
print(torch.cuda.is_available())


MODELS = {0: 'FPT', 1: 'UFPT', 2: 'NPT'}
Dataset = {0: 'FBIRN', 1: 'COBRE', 2: 'OASIS', 3: 'ABIDE'}
gain = {'COBRE': [0.05, 0.65, 0.75], 'FBIRN': [0.85, 0.4, 0.35], 'ABIDE': [0.3, 0.35, 0.8],
        'OASIS': [0.4, 0.65, 0.35]}

Directories = { 'COBRE': 'COBRESaliencies','FBIRN' : 'FBIRNSaliencies', 'ABIDE' : 'ABIDESaliencies','OASIS' : 'OASISSaliencies'}

FNCDict = {"FBIRN": LoadFBIRN, "COBRE": LoadCOBRE, "OASIS": LoadOASIS, "ABIDE": LoadABIDE}

Params = {'FBIRN':[311, 140, 121], 'COBRE': [157, 140, 121], 'OASIS': [372, 120, 101], 'ABIDE': [569, 140, 121]}

train_lim = { 'FBIRN': 250, 'COBRE': 100, 'OASIS': 300, 'ABIDE': 400}

Params_subj_distr = { 'FBIRN': [15, 25, 50, 75, 100], 'COBRE': [15, 25, 40], 'OASIS': [15, 25, 50, 75, 100, 120], 'ABIDE': [15, 25, 50, 75, 100, 150]}

test_lim_per_class = { 'FBIRN': 32, 'COBRE': 16, 'OASIS': 32, 'ABIDE': 50}

# key : value = window_shift : dict for different datasets
sequence_size = { 10: {'FBIRN': 13, 'COBRE': 13, 'OASIS': 11, 'ABIDE': 13}, 1: {'FBIRN': 121, 'COBRE': 121, 'OASIS': 101, 'ABIDE': 121}}

# We need to set gains for the ST-DIM found earlier by Usman.

Params_best_gains = {10: {'FBIRN': {'NPT':[0.15, 0.25, 0.5, 0.9, 0.9], 'UFPT': [0.05, 0.05, 0.05, 0.05, 0.05]}, 'ABIDE': {'NPT': [0.4, 0.5, 0.75, 0.75, 0.45, 0.8], 'UFPT': [0.15, 0.2, 0.25, 0.2, 0.35, 0.35]}, 'OASIS': {'NPT': [0.25, 0.35, 0.75, 0.30, 0.4, 0.35], 'UFPT': [0.05, 0.15, 0.3, 0.35, 0.65, 0.65]}, 'COBRE': {'NPT': [0.25, 0.35, 0.75], 'UFPT': [0.05, 0.45, 0.65]}}, 20: {'COBRE': {'NPT': [0.25, 0.35, 0.75], 'UFPT': [0.05, 0.45, 0.65]}}}
                     

data= int(sys.argv[1])  # Choose FBIRN (0)/ COBRE (1) / OASIS (2) / ABIDE (3)
mode = int(sys.argv[2])  # Choose FPT (0) / UFPT (1) / NPT (2)
window_shift = int(sys.argv[3])
test_ID = int(sys.argv[4])
                 
lr = 0.0005

finalData, FNC, all_labels = FNCDict[Dataset[data]]()


print('Final Data Shape:', finalData.shape)

no_good_comp = 53
sample_y = 20
subjects = Params[Dataset[data]][0]
tc = Params[Dataset[data]][1]
samples_per_subject = sequence_size[window_shift][Dataset[data]] # Params[Dataset[data]][2]

AllData = np.zeros((subjects, samples_per_subject, no_good_comp, sample_y))

for i in range(subjects):
    for j in range(samples_per_subject):
        AllData[i, j, :, :] = finalData[i, :, (j * window_shift):(j * window_shift) + sample_y]

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

        
X = AllData
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
X_test = X[test_index, :, :, :]
Y_test = Y[test_index.long()]
                         
X_sal = X
Y_sal = Y
              
X_test = torch.from_numpy(X_test).float().to(device)
Y_test = torch.from_numpy(Y_test).long().to(device)

X_sal = torch.from_numpy(X_sal).float().to(device)
Y_sal = torch.from_numpy(Y_sal).long().to(device)

              
# X_full = torch.from_numpy(X).float().to(device)
# Y_full = torch.from_numpy(Y).long().to(device)

print('Test Shape:', X_test.shape)
print(X_sal.shape)

dataLoaderTest = get_data_loader(X_test, Y_test, X_test.shape[0])
               
# g = GAIN[ID]

accMat = np.zeros([len(subjects_per_group), Trials])
aucMat = np.zeros([len(subjects_per_group), Trials])

start_time = time.time()


print(f'Allocated: {torch.cuda.memory_allocated()}')

Best_gain = Params_best_gains[10][Dataset[data]][MODELS[mode]]
print('Best Gain Chosen:', Best_gain)

dir = MODELS[mode]   # NPT or UFPT
wdb = 'wandb'
wpath = os.path.join('../', wdb)
sbpath = os.path.join(wpath, 'Sequence_Based_Models')

model_path = os.path.join(sbpath, Directories[Dataset[data]], dir)

for i in range(len(subjects_per_group)):
    for restart in range(Trials):

        samples = subjects_per_group[i]

        g = Best_gain[i]

        HC_random = trials_HC[restart][:samples]  
        SZ_random = trials_SZ[restart][:samples]
        

        HC_index_tr = total_HC_index_tr[HC_random]
        SZ_index_tr = total_SZ_index_tr[SZ_random]


        tr_index = torch.cat((HC_index_tr, SZ_index_tr))
        tr_index = tr_index.view(tr_index.size(0))
        X_train = X[tr_index, :, :, :]
        Y_train = Y[tr_index.long()]

        X_train = torch.from_numpy(X_train).float().to(device)
        Y_train = torch.from_numpy(Y_train).long().to(device)

        print('Train Data Shape:', X_train.shape)
        
#         HC_val_random = trials_HC[restart][samples:]  
#         SZ_val_random = trials_SZ[restart][samples:]
        
#         HC_index_val = total_HC_index_tr[HC_val_random]
#         SZ_index_val = total_SZ_index_tr[SZ_val_random]
        
#         val_index = torch.cat((HC_index_val, SZ_index_val))
#         val_index = val_index.view(val_index.size(0))
#         X_val = X[val_index, :, :, :]
#         Y_val = Y[val_index.long()]

#         X_val = torch.from_numpy(X_val).float().to(device)
#         Y_val = torch.from_numpy(Y_val).long().to(device)
        
#         dataLoaderVal = get_data_loader(X_val, Y_val, X_val.shape[0])

        
        
        np.random.seed(0)
        randomize = np.random.permutation(X_train.shape[0])

        X_train_go = X_train[randomize]
        Y_train_go = Y_train[randomize]

        dataLoaderTrain = get_data_loader(X_train_go, Y_train_go, 32)
        
        dataLoaderTrainCheck = get_data_loader(X_train_go, Y_train_go, X_train_go.shape[0])
        
        print(f'Test Split Starts: {test_ID}')

        print('MILC + with TOP Anchor + both uniLSTM')
        print(f'Model Started: {restart}\nSPC: {samples}\nGain: {g}\nExperiment MODE: {MODELS[mode]}\nDataset: {Dataset[data]}')

        encoder = NatureOneCNN(X.shape[2], args)
        
        if MODELS[mode] == 'UFPT':
            modelpath = './stdim_encoder.pt'
            model_dict = torch.load(modelpath, map_location=device) 
            encoder.load_state_dict(model_dict)
            print("STDIM PRE-TRAINED ENCODER LOADED...")
        else:
            print("NO PRE-TRAINING IS NEEDED FOR THE ENCODER...")
        
        
        model = LSTM(encoder, 256, 200, 2, g).float()

        
        print(f'Allocated: {torch.cuda.memory_allocated()}')
        

        model.to(device)
        optimizer, accMat[i, restart], aucMat[i, restart] = train_model(model, dataLoaderTrain, dataLoaderTrainCheck, dataLoaderTest, 100, lr)

#         dataLoaderFull = get_data_loader(X_sal, Y_sal, X_sal.shape[0])          
        dataLoaderSal = get_data_loader(X_sal, Y_sal, 1)

        middle_time = time.time() - start_time
        print('Total Time for Training:', middle_time)


        prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_simple_gain_{g}_window_shift_{window_shift}_{dir}_test_id_{test_ID}_STDIM_May9_lr_{lr}'
        
#         prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_gain_{g}_window_shift_{window_shift}_{dir}_arch_chk_LSTM_milc'
        

        # Re-loading the models

#         model = save_reload_model(model, model_path, prefix, device, restart, save_reload = 'reload')
        

        # Save model
#         save_reload_model(model, model_path, prefix, device, restart, save_reload = 'save')

        if torch.cuda.is_available():
            
            print(f'Allocated: {torch.cuda.memory_allocated()}')
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Allocated: {torch.cuda.memory_allocated()}')
            
        
basename2 = os.path.join(sbpath, Directories[Dataset[data]], dir)

prefix = f'{Dataset[data]}_all_spc_simple_val_best_gains_window_shift_{window_shift}_{dir}_test_id_{test_ID}_STDIM_May9_lr_{lr}'

# prefix = f'{Dataset[data]}_all_spc_all_gains_{GAIN[ID]}_window_shift_{window_shift}_{dir}_STDIM'

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
