# This is LSTM and attention only method for encoding...
# Proposed as an alternative of the original CNN encoder to get more stable saliency maps
# some weight normalization applied
# WE USED THIS PRETRAINING CODE FOR THE RESULTS WE REPORTED IN SALIENCY PAPER


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import RandomSampler, BatchSampler

from source.utils import get_argparser
from source.encoders_ICA import NatureCNN, ImpalaCNN, NatureOneCNN, LinearEncoder
from myscripts.LoadRealData import LoadHCP, LoadHCPStride10, LoadHCPStride20, LoadHCPFull
from myscripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2, \
    artificial_batching_trend, multi_way_data, my_multi_block_var_data, actual_spatial, three_class

from source.ATTLSTM import subjLSTM, onlyLSTM
from torch.nn.utils import weight_norm as wn

import sys
import os
import time

run = int(sys.argv[1])

# parser = get_argparser()
# args = parser.parse_args()

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
args = Namespace(end_with_relu=False, feature_size=256, no_downsample=True, script_ID = run)



start_time = time.time()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# np.random.seed(run)

def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index

class LSTM(torch.nn.Module):

    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, learning_rate, gain):
        super(LSTM, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = hidden_nodes
        
        self.enc_out = input_size
        self.lstm = nn.LSTM(input_size, hidden_nodes, batch_first=True)
        
        # input size for the top lstm is the hidden size for the lower
        
        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)

        # New version - 2
#         self.encoder = nn.LSTM(enc_input_size, self.enc_out // 2, batch_first = True, bidirectional = True)
        
        # Previous version

#         self.attnenc = nn.Sequential(
#              wn(nn.Linear(self.enc_out, 64)),   
#              wn(nn.Linear(64, 1))
#         )
        
        # New Version 1, Anchoring
        self.attnenc = nn.Sequential(
            nn.Linear(2*self.enc_out, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # New Version 2, No anchoring
        
#         self.attnenc = nn.Sequential(
#             nn.Linear(self.enc_out, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

        self.classifier1 = nn.Linear(input_size, hidden_nodes)
        
        # Previous version
        
#         self.attn = nn.Sequential(
#             wn(nn.Linear(2*self.hidden, 128)),
#             wn(nn.Linear(128, 1))
#         )
        
        # New version both
        
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.gain = gain 
        self.init_weight()

        self.optimizer = torch.optim.Adam(list(self.attn.parameters()) + list(self.lstm.parameters()) + list(self.encoder.parameters()) + list(self.attnenc.parameters()) + list(self.classifier1.parameters()), lr=learning_rate, eps=1e-5)
        
    def init_weight(self):
        
        print('Initializing')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain= self.gain)
        for name, param in self.attnenc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain= self.gain)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain= self.gain)
        for name, param in self.classifier1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain= self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain= self.gain)
        
    def generate_batch(self, episodes, mode):

        if mode == 'train':
            BS = 32
        else:
            BS = len(episodes)

        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=False),
                               32, drop_last=False)

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            ts_number = torch.LongTensor(indices)
        
            sx = []
           
            for episode in episodes_batch:
                # Get all samples from this episode
                mean = episode.mean()
                sd = episode.std()
                episode = (episode - mean) / sd
                sx.append(episode)

            yield torch.stack(sx).to(device) , ts_number.to(device)
            
    def init_hidden(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        
        return (h0, c0)
    
    def init_hidden_enc(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.enc_out, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.enc_out, device=device))
        
        return (h0, c0)


    def forward(self, epoch, episodes, mode):
        
        epoch_loss, steps= 0., 0
        epoch_accuracy = 0.
        
        data_generator = self.generate_batch(episodes, mode)

        for sx, ts_number in data_generator:
           
            loss = 0.
            accuracy = 0.

            b_size = sx.size(0)
            s_size = sx.size(1)
            sx = sx.view(-1, sx.shape[2], 20)
            sx = sx.permute(0, 2, 1)
            
            enc_batch_size = sx.size(0)
            
            self.enc_hidden = self.init_hidden_enc(enc_batch_size, device)
            inputs_tensor, self.enc_hidden = self.encoder(sx, self.enc_hidden)
            
            inputs_tensor = self.get_attention_enc(inputs_tensor)
            inputs_tensor = inputs_tensor.view(b_size, s_size, -1)

            self.lstm_hidden = self.init_hidden(b_size, device)
            outputs, self.lstm_hidden = self.lstm(inputs_tensor, self.lstm_hidden)

            N = inputs_tensor.size(0)
            targets = torch.arange(N).to(device)
            
            logits = self.get_attention(outputs)
            logits = logits.to(device)
            sx = inputs_tensor.size(0)
            sy = inputs_tensor.size(1)
            v = np.arange(0, sx)
#             random_matrix = torch.randperm(sy)   # Usman used for random window
            random_matrix = np.arange(sy)
            for loop in range(sx - 1):
#                 random_matrix = np.concatenate((random_matrix, torch.randperm(sy)), axis=0) # for random window
                random_matrix = np.concatenate((random_matrix, np.arange(sy)), axis=0)
            random_matrix = np.reshape(random_matrix, (sx, sy))
            for y in range(sy):
                
                y_index = random_matrix[:, y]

                positive = inputs_tensor[v, y_index, :].clone()
                positive = self.classifier1(positive)
                mlogits = torch.matmul(positive, logits.t())
                mlogits = mlogits.to(device)
                step_loss = F.cross_entropy(mlogits, targets)
                sig = torch.softmax(mlogits, dim=1).to(device)
                
                _, preds = torch.max(sig.data, 1)
                step_acc = (preds == targets).sum().item()
                step_acc = step_acc / N

                loss += step_loss
                accuracy += step_acc
            loss = loss / (sy)
            accuracy = accuracy / (sy)
            
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_accuracy += accuracy


            steps += 1
        
        return epoch_loss/steps, epoch_accuracy/steps


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
#         logits = self.decoder(attn_applied)
        return attn_applied

    def get_attention_enc(self, outputs):
        
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        

        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.enc_out)

        weights = self.attnenc(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        
#         print('At attention in enc weights:', weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()

        # Pass the weighted output to decoder
#         logits = self.decoder(attn_applied)
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


def train_model(model, X_train, X_test, epochs):
#     loss = torch.nn.CrossEntropyLoss()

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, 'min')

    for epoch in range(epochs):
        
        train_loss, train_acc = model(epoch, X_train, "train")
        
        test_loss, test_acc = model(epoch, X_test, "test")
        
        
        print("epoch: " + str(epoch) + " train loss: " + str(train_loss) + ", train acc: " + str(train_acc) + " test loss: " + str(test_loss)+", test acc: " + str(test_acc))
        
        scheduler.step(test_loss)



print(torch.cuda.is_available())


# A = np.load("./matrix_bg.npy")
# B = np.load("./matrix_fg1.npy")
# C = np.load("./matrix_fg2.npy")


# Data, labels, start_positions = artificial_batching_trend(A, B, C, 40000, 140, A.shape[0], p_steps=20, alpha=0, seed=1950)
# Data, labels, start_positions, masks = multi_way_data(40000, 140, 50, seed=1988)

# Data, labels, start_positions, masks = artificial_batching_patterned_space2(40000, 140, 50, seed=1901)

# Data, labels, start_positions, masks = actual_spatial(40000, 140, 50, seed=1901)


# print(Data[0, 0, 0:20])
# Data = np.moveaxis(Data, 1, 2)    # It needs if encoder used

# print('Original Data Shape:', Data.shape)

# HC_index, SZ_index = find_indices_of_each_class(labels)

# Data = Data[HC_index]
# labels = labels[HC_index]

# print('All Class 0 Shape:', Data.shape)

# subjects = Data.shape[0] 
# sample_x = 50 # A.shape[0]
# sample_y = 20
# tc = 140
# samples_per_subject = 13  
# window_shift = 10

# finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))

# for i in range(subjects):
#     for j in range(samples_per_subject):
#         finalData[i, j, :, :] = Data[i, :, (j * window_shift):(j * window_shift) + sample_y]


# For real data ...

finalData, index_array = LoadHCPStride20()  # Earlier used stride 20 as in Usman's MILC github
print('Data shape ended up with:', finalData.shape)
tr_index = index_array[123:823]
test_index = index_array[0:123]


X = finalData   # it needs for the encoder
 

print(X.shape)




# X_train = X[:16000]

X_train = X[tr_index]

X_train = torch.from_numpy(X_train).float().to(device)


# X_test = X[18000:]

X_test = X[test_index]
             
X_test = torch.from_numpy(X_test).float().to(device)


print(X_train.shape)
print(X_test.shape)

               
# gain = 1.4 # for old two class data, I used 1.0 and 0.1
Gain = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0}

gain = Gain[run]

model = LSTM(X.shape[2], 256, 200, X.shape[1], 2, 0.005, gain).float()

print('MILC + with Anchor + both uniLSTM')

print('The gain value used:', gain)
train_model(model, X_train, X_test, 150)
         

middle_time = time.time() - start_time
print('Total Time for Training:', middle_time)


Maindir = "InterpolationVARDataEncoder"
dir = "HCP"
wdb = 'wandb'
wpath = os.path.join('../', wdb)
sbpath = os.path.join(wpath, 'SynDataPretrainedEncoder')
model_path = os.path.join(sbpath, Maindir, dir)

# Re-loading the models

# print('Loading model for check...')
# model_path = os.path.join(model_path, 'pretrained_milc_model_'+str(args.script_ID)+'.pt')
# model_dict = torch.load(model_path, map_location=device)  # with good components
# model.load_state_dict(model_dict)
# model.to(device)

# train_model(model, X_train, X_test, 10, .0003)
         


# Save pre-trained model

model_file_path = os.path.join(model_path, 'HCP_LSTM_only_gain_'+str(gain)+'_May032021_Encoder_Anchoring_'+str(args.script_ID) + '.pt')


# model_file_path = os.path.join(model_path, 'HCP_LSTM_only_gain_'+str(gain)+'_Nov302020_'+str(args.script_ID) + '.pt')

torch.save(model.state_dict(), model_file_path)
print(model_file_path)

elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
