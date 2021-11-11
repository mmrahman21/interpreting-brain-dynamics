'''
This code is used to compute saliency with the stored models.
The models were trained in low data regime using another script. 
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from myscripts.milc_utilities_handler import load_pretrain_model, get_captum_saliency, get_captum_saliency_norm_data, save_predictions, save_reload_model,save_acc_auc


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


GAIN = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0}  

ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad', 4:'', 5:'smoothgrad', 6:'smoothgrad_sq', 7: 'vargrad'}

saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime'}


start_time = time.time()

device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

print(device)


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


class LSTM(torch.nn.Module):

    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, gain):
        super(LSTM, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = hidden_nodes
        
        self.enc_out = input_size
        self.lstm = nn.LSTM(input_size, hidden_nodes, batch_first=True)
        
        # input size for the top lstm is the hidden size for the lower
        
        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)  

        self.attnenc = nn.Sequential(
             nn.Linear(2*self.enc_out, 64),
             nn.ReLU(),
             nn.Linear(64, 1)
        )
     
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 200),
            nn.ReLU(),
            nn.Linear(200, output_size),
            nn.Sigmoid()
        )
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing All components')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attnenc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
            print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
                
    def init_hidden(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        
        return (h0, c0)
    
    def init_hidden_enc(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.enc_out, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.enc_out, device=device))
        
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
        x = x.permute(0, 2, 1)
        
        enc_batch_size = x.size(0)
            
        self.enc_hidden = self.init_hidden_enc(enc_batch_size, device)
        out, self.enc_hidden = self.encoder(x, self.enc_hidden)

        out = self.get_attention_enc(out)
        out = out.view(b_size, s_size, -1)

        self.lstm_hidden = self.init_hidden(b_size, device)
        lstm_out, self.lstm_hidden = self.lstm(out, self.lstm_hidden)

        lstm_out = self.get_attention(lstm_out)
        
        lstm_out = lstm_out.view(b_size, -1)
        
        smax = torch.nn.Softmax(dim=1)
        lstm_out_smax = smax(lstm_out)
        
        return lstm_out  #lstm_out_smax

    def get_attention(self, outputs):
        
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.hidden)

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
    
    # Do not make the shuffle = True during saliency computations
    
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size)

    return dataLoader


def train_model(model, loader_train, loader_train_check, loader_test, epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model.cuda()
    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):

        for i, data in enumerate(loader_train):
            x, y = data
            # x = x.permute(1, 0, 2)
            optimizer.zero_grad()
            outputs= model(x)
           
            l = loss(outputs, y)
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
        
        print("epoch: " + str(epoch) + ", loss: " + str(l.detach().item()) +", test acc: " + str(accuracy / y_test.size(0)) + ", roc: " + str(roc) +", train acc: " + str(train_accuracy) +" , train roc: " + str(train_roc))
        
        test_loss = loss(outputs, y_test)
        scheduler.step(test_loss)
        
        
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

# These gains were based on cross-validation based search

# Params_best_gains = {1: {'FBIRN': {'NPT':[1.0, 0.9, 0.9, 1.0, 1.0], 'UFPT': [1.0, 0.8, 0.7, 0.5, 0.4]}, 'OASIS': {'NPT': [0.9, 0.9, 1.0, 1.1, 1.1, 1.0], 'UFPT': [0.8, 0.8, 0.4, 0.3, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.0, 0.9, 0.9, 1.0, 0.7, 0.6], 'UFPT': [0.4, 0.8, 0.3, 0.2, 0.2, 0.1]}, 'COBRE': {'NPT': [1.1, 0.7, 0.8], 'UFPT': [1.6, 1.3, 1.3]}}, 
#                      10: {'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [1.2, 1.2, 1.4]}},
#                      20: {'COBRE': {'NPT': [1.5, 0.8, 0.8], 'UFPT': [1.2, 1.3, 1.3]}}  
#                     }

# Data normalize and cross-validation search (cross validation on training for FBIRN)
# simple gain selection (used only one fold for ABIDE) 

# Params_best_gains = {1: {'FBIRN': {'NPT':[0.9, 0.9, 0.8, 0.7, 0.6], 'UFPT': [0.8, 0.8, 0.6, 0.7, 0.6]}, 'ABIDE': {'NPT': [0.9, 0.5, 0.9, 1.0, 1.0, 0.7], 'UFPT': [1.0, 0.9, 0.4, 0.4, 0.1, 0.2]}, 'OASIS': {'NPT': [0.6, 0.9, 1.0, 0.7, 0.9, 0.6], 'UFPT': [0.5, 0.4, 0.3, 0.6, 1.0, 1.1]},'COBRE': {'NPT': [1.0, 0.3, 0.5], 'UFPT': [0.7, 0.6, 1.2]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}

# Revised h_fixed Simiple Gain Selection

Params_best_gains = {1: {'FBIRN': {'NPT':[1.2, 1.3, 0.7, 1.1, 0.9], 'UFPT': [0.9, 0.7, 0.6, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.3, 1.3, 1.1, 0.9, 1.3, 0.6], 'UFPT': [0.4, 0.6, 0.5, 0.5, 0.3, 1.2]}, 'OASIS': {'NPT': [0.1, 1.2, 1.0, 1.0, 1.2, 0.7], 'UFPT': [0.7, 0.7, 1.3, 0.6, 1.1, 1.2]},'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [0.4, 1.1, 1.3]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}
                     

data= int(sys.argv[1])  # Choose FBIRN (0)/ COBRE (1) / OASIS (2) / ABIDE (3)
mode = int(sys.argv[2])  # Choose FPT (0) / UFPT (1) / NPT (2)
test_ID = int(sys.argv[3]) # select the test split

finalData, FNC, all_labels = FNCDict[Dataset[data]]()


print('Final Data Shape:', finalData.shape)

no_good_comp = 53
sample_y = 20
subjects = Params[Dataset[data]][0]
tc = Params[Dataset[data]][1]
samples_per_subject = Params[Dataset[data]][2]
window_shift = 1

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


# for test_ID in range(1):
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


print('Test Shape:', X_test.shape)
print(X_sal.shape)

dataLoaderTest = get_data_loader(X_test, Y_test, X_test.shape[0])

# ID = int(sys.argv[3])
# g = GAIN[ID]

accMat = np.zeros([len(subjects_per_group), Trials])
aucMat = np.zeros([len(subjects_per_group), Trials])

start_time = time.time()


print(f'Allocated: {torch.cuda.memory_allocated()}')

Best_gain = Params_best_gains[window_shift][Dataset[data]][MODELS[mode]]
print('Best Gain Chosen:', Best_gain)

dir = MODELS[mode]   # NPT or UFPT
wdb = 'wandb'
wpath = os.path.join('../', wdb)
sbpath = os.path.join(wpath, 'Sequence_Based_Models')

model_path = os.path.join(sbpath, Directories[Dataset[data]], dir)

for saliency_id in range(8):

    for i in range(len(subjects_per_group)):
        for restart in range(Trials):

            samples = subjects_per_group[i]

            g = Best_gain[i]

    #         HC_random = trials_HC[restart][:samples]  
    #         SZ_random = trials_SZ[restart][:samples]


    #         HC_index_tr = total_HC_index_tr[HC_random]
    #         SZ_index_tr = total_SZ_index_tr[SZ_random]


    #         tr_index = torch.cat((HC_index_tr, SZ_index_tr))
    #         tr_index = tr_index.view(tr_index.size(0))
    #         X_train = X[tr_index, :, :, :]
    #         Y_train = Y[tr_index.long()]

    #         X_train = torch.from_numpy(X_train).float().to(device)
    #         Y_train = torch.from_numpy(Y_train).long().to(device)

    #         print('Train Data Shape:', X_train.shape)


    #         np.random.seed(0)
    #         randomize = np.random.permutation(X_train.shape[0])

    #         X_train_go = X_train[randomize]
    #         Y_train_go = Y_train[randomize]

    #         dataLoaderTrain = get_data_loader(X_train_go, Y_train_go, 32)

    #         dataLoaderTrainCheck = get_data_loader(X_train_go, Y_train_go, X_train_go.shape[0])

            print(f'\n\nComputing Saliencies....\nTest Split Starts: {test_ID}')

            print('MILC + with TOP Anchor + both uniLSTM')
            print(f'Model Started: {restart}\nSPC: {samples}\nGain: {g}\nExperiment MODE: {MODELS[mode]}\nDataset: {Dataset[data]}')

            model = LSTM(X.shape[2], 256, 200, 121, 2, g).float()


            print(f'Allocated: {torch.cuda.memory_allocated()}')

    #         model = load_pretrain_model(model, exp_type = MODELS[mode], device = device)


            model.to(device)
    #         optimizer, accMat[i, restart], aucMat[i, restart] = train_model(model, dataLoaderTrain, dataLoaderTrainCheck, dataLoaderTest, 200, .0005)

    #         dataLoaderFull = get_data_loader(X_sal, Y_sal, X_sal.shape[0])          
            dataLoaderSal = get_data_loader(X_sal, Y_sal, 1)

            middle_time = time.time() - start_time
            print('Total Time for Training:', middle_time)

            prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_simple_gain_{g}_window_shift_{window_shift}_{dir}_arch_chk_h_fixed_test_id_{test_ID}_LSTM_milc'
    #         prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_gain_{g}_window_shift_{window_shift}_{dir}_arch_chk_h_fixed_LSTM_milc'

            # Re-loading the models

            model = save_reload_model(model, model_path, prefix, device, restart, save_reload = 'reload')


            # Compute Saliency
            saliencies = get_captum_saliency_norm_data(model, dataLoaderSal, saliency_id, device)

            basename = os.path.join(sbpath, Directories[Dataset[data]], dir, 'Saliency')
            path1 = os.path.join(basename, prefix+'_prediction'+'_'+saliency_options[saliency_id]+'_norm_data_'+str(restart))
            np.save(path1, saliencies)

            print("Saliency saved here:", path1)


            # Save predictions
    #         prediction_save_path = os.path.join(basename, prefix+'_labels_save_'+str(restart))   
    #         save_predictions(X, model, dataLoaderFull, prediction_save_path)


            # Save model
    #         save_reload_model(model, model_path, prefix, device, restart, save_reload = 'save')

            if torch.cuda.is_available():

                print(f'Allocated: {torch.cuda.memory_allocated()}')
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print(f'Allocated: {torch.cuda.memory_allocated()}')

        
elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())
