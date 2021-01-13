'''
This code is used to run experiments in low data regime
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

from datetime import datetime
import pandas as pd

import sys
import os
import time

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion, 
    Saliency,
    GuidedBackprop,
)


GAIN = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0}  

SCALES = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6}

# scale = int(sys.argv[1])

# ID = int(sys.argv[1])
# Saliency_ID = int(sys.argv[1])   

Ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad', 4:'', 5:'smoothgrad', 6:'smoothgrad_sq', 7: 'vargrad'}

saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG'}

# 'smoothgrad': 'SG' , 'smoothgrad_sq': 'SGSQ', 'vargrad': 'VG', '':''}


start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# np.random.seed(run)

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
             nn.Linear(self.enc_out, 64),
             nn.Linear(64, 1)
        )
     
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.Linear(128, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 200),
            nn.Linear(200, output_size)
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

    def forward(self, x):
        
        b_size = x.size(0)
        s_size = x.size(1)
        x = x.view(-1, x.shape[2], 20)
        x = x.permute(0, 2, 1)
        
        
        out, hidden = self.encoder(x)
        out = self.get_attention_enc(out)
        out = out.view(b_size, s_size, -1)
        lstm_out, hidden = self.lstm(out)
        

        # lstm_out = self.lstm(x)

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
#         B= outputs[:,-1, :]
#         B = B.unsqueeze(1).expand_as(outputs)
#         outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs.reshape(-1, self.enc_out)

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
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size)

    return dataLoader


def train_model(model, loader_train, loader_test, epochs, learning_rate):
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
        
        print("epoch: " + str(epoch) + ", loss: " + str(l.detach().item()) +", train acc: " + str(accuracy / y_test.size(0)) + ", roc: " + str(roc))
        
        test_loss = loss(outputs, y_test)
        scheduler.step(test_loss)
        
        
    return optimizer, accuracy / y_test.size(0), roc


def get_saliency(model, loaderSal):
    
    loss = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    # model.eval()
    for param in model.parameters():
        param.requires_grad = False
    saliencies1 = []
    saliencies2 = []
    outs = []
    
    accuracy = 0
    count = 0
    for i, data in enumerate(loaderSal):
        count += 1
        if count > 10000:
            break
        if i % 1000 == 0:
            print(i)
        x, y = data
        # x = x.permute(1, 0, 2)
        x = x.to(device)
        y = y.to(device)
      
        x.requires_grad_()

        output = model(x)
        
#         output = output.unsqueeze(0)

        l = loss(output, y)
        _, pred = torch.max(output.data, 1)
        accuracy = accuracy + (pred == y).sum().item()
        
        grad_outputs = output
#         grad_outputs = torch.ones(x.shape[0], 2).to(device)
# #         grad_outputs[:, y] = 1

#         grad_outputs[:, pred] = 1


#        # Set retain_graph = True to continue gradient calc on other variable as well
        
#         output.backward(gradient=grad_outputs, retain_graph=True)  

#         grads = x.grad     
       
#         saliency = np.squeeze(grads.cpu().numpy())
#         saliencies1.append(saliency)
        
#         x.grad.zero_()

#         # Saliency with respect to incorrect class

#         grad_outputs = torch.zeros(x.shape[0], 2).to(device)
# #         grad_outputs[:, 1-y] = 1
        
#         grad_outputs[:, 1-pred] = 1
        
        output.backward(gradient=grad_outputs)

#         l.backward()
        
        grads = x.grad
        
        saliency = np.squeeze(grads.cpu().numpy())
        saliencies2.append(saliency)
        outs.append(np.squeeze(output.cpu().detach().numpy()))
        
    print('Accuracy:', accuracy/count)
        
    return saliencies2, outs

def get_captum_saliency(model, loaderSal):
#     model.eval()
#     model.cpu()
#     device = "cpu"
    model.zero_grad()
    
    if Saliency_ID <4:
        sal = Saliency(model)
        
    else:
        sal = IntegratedGradients(model)
        
#     sal = GuidedBackprop(model)

    if Saliency_ID !=0 and Saliency_ID !=4:
        nt = NoiseTunnel(sal)
        print('Ensemble Begins .. with', saliency_options[Saliency_ID])

    print('Ensembles:', Ensembles[Saliency_ID])
    
    saliencies = []
    for i, data in enumerate(loaderSal):
        if i % 1000 == 0:
            print(i)
        x, y = data   
        
        outputs = model(x)
        _, preds = torch.max(outputs.data, 1)
#         print('True ={}, Predicted= {}'.format(y.item(), preds.item()))
        
        bl = torch.zeros(x.shape).to(device)
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        if Saliency_ID == 0:
            S = sal.attribute(x, target=preds, abs=False)
            
        elif Saliency_ID == 4:
            S = sal.attribute(x,bl,target=preds)
            
        elif Saliency_ID > 0 and Saliency_ID < 4:
            S = nt.attribute(x, nt_type=Ensembles[Saliency_ID], n_samples=10, target=preds, abs=False)
            
        else:
            S = nt.attribute(x, nt_type=Ensembles[Saliency_ID], n_samples=10, baselines = bl, target=preds)         

        saliencies.append(np.squeeze(S.cpu().detach().numpy()))
             
    return saliencies


def LoadUFPT(model):
#     modelpath = '../wandb/SynDataPretrainedEncoder/InterpolationVARDataEncoder/HCP/HCP_LSTM_only_gain_0.1_Aug062020_1.pt'
    modelpath = '../wandb/SynDataPretrainedEncoder/InterpolationVARDataEncoder/HCP/HCP_LSTM_only_gain_0.1_Nov302020_1.pt'
    modelA_dict = torch.load(modelpath, map_location=device)  # Pre-trained model is model A
    
    print('Model loaded from here:', modelpath)


    modelB_dict = model.state_dict()    # Let's assume downstream model as B

    print("modelB (downstream model) is going to use the common layers parameters from modelA")
    pretrained_dict = modelA_dict
    model_dict = modelB_dict

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model.to(device)

    print('PRE-TRAINED MODELS LOADED')
    return model 

def LoadNPT(model):
    print('NO MODEL LOADING IS NEEDED , IT WILL BE TRAINED AFRESH...')
    return model


def weight_display(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
   
    if isinstance(m, nn.Linear):
        print('Displaying Linear Layer')

        print(m.weight.data.shape)
        print(m.weight.data.std())
        print(m.weight.data.mean())

        print('Bias:', m.bias.data.shape)
        print('Bias:', m.bias.data.std())
        print('Bias:', m.bias.data.mean())
        
    elif isinstance(m, nn.LSTM):

        print('Displaying LSTM')
        for param in m.parameters():
            print(param.data.shape)
            print(param.data.std())
            print(param.data.mean())

def weight_rescale(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_rescale)
        
        seeing the distribution of pre-trained weights, it sounds similar to fresh initializations...
        so no further scaling is needed as decided. 
    '''
   
    if isinstance(m, nn.Linear):
        print('Modified Rescaling Linear Layer')
        m.weight.data = SCALES[scale]*(m.weight.data)
        m.bias.data = SCALES[scale]*(m.bias.data)
        
    elif isinstance(m, nn.LSTM):

        print('Modified Rescaling LSTM')
        for param in m.parameters():
            param.data = SCALES[scale]*(param.data)

def weight_rescale_norm(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_rescale)
        
        seeing the distribution of pre-trained weights, it sounds similar to fresh initializations...
        so no further scaling is needed as decided. 
    '''
   
    if isinstance(m, nn.Linear):
        print('Normalizing Linear Layer')
#         m.weight.data = GAIN[ID]*(m.weight.data)
        x = (m.weight.data - m.weight.data.mean())/m.weight.data.std()
        m.weight.data = x
#         m.bias.data = GAIN[ID]*(m.bias.data)
        
    elif isinstance(m, nn.LSTM):

        print('Normalizing LSTM')
        for param in m.parameters():
            x = (param.data - param.data.mean())/param.data.std()
            param.data = x

            
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

test_lim_per_class = { 'FBIRN': 32, 'COBRE': 15, 'OASIS': 32, 'ABIDE': 50}


mode = 2  # Choose FPT (0) / UFPT (1) / NPT (2)
data= 0  # int(sys.argv[1])  # Choose FBIRN (0)/ COBRE (1) / OASIS (2) / ABIDE (3)

# current_gain = 0.1 # gain[Dataset[data]][list(MODELS.keys())[list(MODELS.values()).index(MODELS[mode])]]  # UFPT - 1, NPT - 2

# print("Current Gain:", current_gain)


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

total_HC_index_tr = HC_index[:len(HC_index) - test_lim_per_class[Dataset[data]]]
total_SZ_index_tr = SZ_index[:len(SZ_index) - test_lim_per_class[Dataset[data]]]

print('Length of training HC:', len(total_HC_index_tr))
print('Length of training SZ:', len(total_SZ_index_tr))

HC_index_test = HC_index[len(HC_index) - (test_lim_per_class[Dataset[data]]):]
SZ_index_test = SZ_index[len(SZ_index) - (test_lim_per_class[Dataset[data]]):]

        
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

Best_gain = [1.6, 0.8, 0.3, 0.9, 0.5]  # For FBIRN NPT
# Best_gain = [0.4, 0.1, 0.3, 0.1, 0.1]  # For FBIRN UFPT

# Best_gain = [1.4, 0.7, 0.7]  # For COBRE NPT
# Best_gain = [1.5, 0.9, 1.7]  # For COBRE UFPT

# Best_gain = [1.3, 1.2, 1.6, 1.0, 1.3, 0.9] # For OASIS NPT
# Best_gain = [1.3, 0.5, 0.8, 0.3, 0.3, 0.3] # For OASIS UFPT

# Best_gain = [0.8, 0.9, 0.9, 1.0, 0.9, 0.8] # For ABIDE NPT
# Best_gain = [0.5, 0.7, 0.2, 0.8, 0.3, 0.1] # For ABIDE UFPT


for i in range(len(subjects_per_group)):
    for restart in range(Trials):

        samples = subjects_per_group[i]

        g = Best_gain[i]

        print('Using SPC:', samples)

        print('Gain:', g)

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

        dataLoaderTrain = get_data_loader(X_train, Y_train, 32)

        print('Model {} started'.format(restart))

        model = LSTM(X.shape[2], 256, 200, 121, 2, g).float()

        print('MILC + with TOP Anchor + both uniLSTM')
        print('Gain value used:', g)

    #         print('Scaling Factor:', SCALES[scale])

        print("Experiment MODE:", MODELS[mode])
        print("Dataset :", Dataset[data])


        pretrainDict = { "NPT": LoadNPT, "UFPT": LoadUFPT}

#         model = pretrainDict[MODELS[mode]](model)


    #     print('BEFORE WEIGHT RESCALING')
    #     print('-'*60)

    #         model.apply(weight_display)

    #         model.apply(weight_rescale)

    #         model.init_weight()

    #     print('AFTER WEIGHT RESCALING')
    #     print('-'*60)

    #     model.apply(weight_display)


#         model.to(device)
#         optimizer, accMat[i, restart], aucMat[i, restart] = train_model(model, dataLoaderTrain, dataLoaderTest, 200, .0005)

        dataLoaderFull = get_data_loader(X_sal, Y_sal, X_sal.shape[0])          
        dataLoaderSal = get_data_loader(X_sal, Y_sal, 1)

        middle_time = time.time() - start_time
        print('Total Time for Training:', middle_time)


        dir = MODELS[mode]   # NPT or UFPT
        wdb = 'wandb'
        wpath = os.path.join('../', wdb)
        sbpath = os.path.join(wpath, 'Sequence_Based_Models')


#         prefix = Dataset[data]+'_spc_'+str(subjects_per_group[ID])+'_gain_'+str(g)+'_'+'NPT_Revised_LSTM_milc'
        prefix = Dataset[data]+'_spc_'+str(subjects_per_group[i])+'_gain_'+str(g)+'_'+'NPT_LSTM_milc'


        # Re-loading the models

        model_path = os.path.join(sbpath, Directories[Dataset[data]], dir)
        model_path = os.path.join(model_path, prefix+'_captum_use_'+str(restart)+ '.pt')
        model_dict = torch.load(model_path, map_location=device)  # with good components
        model.load_state_dict(model_dict)
        model.to(device)
        print('Model loaded from:', model_path)



    #     saliencies1, saliencies2, outputs = get_saliency(model, dataLoaderSal)

#         saliencies = get_captum_saliency(model, dataLoaderSal)
#         saliencies1 = np.stack(saliencies, axis=0)
#         print(saliencies1.shape)

        basename = os.path.join(sbpath, Directories[Dataset[data]], dir, 'Saliency')
#         path1 = os.path.join(basename, prefix+'_prediction'+'_'+saliency_options[Saliency_ID]+'_'+str(restart))
#         np.save(path1, saliencies1)

#         print("Saliency saved here:", path1)


        # Only for saving predictions and true labels
        predictions = np.zeros((X.shape[0], 2))
        for t, d in enumerate(dataLoaderFull):
            if t % 1000 == 0:
                print(t)
            x, y = d  

            outputs = model(x)
            _, preds = torch.max(outputs.data, 1) 
    #         print(preds)
            accuracy = (preds == y).sum().item()
            predictions[:, 0] = y.cpu().numpy()
            predictions[:, 1] = preds.cpu().detach().numpy()
            print('Acc obtained overall:', accuracy)
            
            sig = F.softmax(outputs, dim=1).to(device)
            y_scores = sig.detach()[:, 1]
            roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
            aucMat[i, restart] = roc
            accMat[i, restart] = accuracy / y.size(0)

#         path2 = os.path.join(basename, prefix+'_labels_save_'+str(restart))

#         np.save(path2, predictions)
#         print('Saving here...', path2)

        with torch.no_grad():
            torch.cuda.empty_cache()


#         # Save model

#         model_path = os.path.join(sbpath, Directories[Dataset[data]], dir)
#         model_path = os.path.join(model_path, prefix+'_captum_use_'+str(restart) + '.pt')
#         torch.save(model.state_dict(), model_path)
#         print("Model saved here:", model_path)

# basename2 = os.path.join(sbpath, Directories[Dataset[data]], dir)
# prefix = Dataset[data]+'_all_spc_'+'gain_'+str(g)+'_UFPT_LSTM_milc'
# # prefix = Dataset[data]+'_all_spc_'+'best_gains'+'_NPT_LSTM_milc'
# accDataFrame = pd.DataFrame(accMat)
# accfname = os.path.join(basename2, prefix +'_ACC.csv')
# accDataFrame.to_csv(accfname)
# print('Result Saved Here:', accfname)


# aucDataFrame = pd.DataFrame(aucMat)
# aucfname = os.path.join(basename2, prefix +'_AUC.csv')
# aucDataFrame.to_csv(aucfname)

print("AUC:", aucMat)
print("ACC:", accMat)

elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())