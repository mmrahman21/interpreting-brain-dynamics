import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout, calculate_accuracy_by_labels, calculate_FP, calculate_FP_Max
from .trainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms
import matplotlib.pylab as plt
import matplotlib.pyplot as pl
import torchvision.transforms.functional as TF
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import csv
import time


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class LSTMTrainer(Trainer):
    def __init__(self, encoder, lstm, config, device, exp, tr_labels, val_labels, test_labels, wandb=None, trial="", crossv="", gtrial=""):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.lstm = lstm
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.val_labels = val_labels
        self.exp = exp
        self.patience = self.config["patience"]
        self.dropout = nn.Dropout(0.65).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.sample_number = config['sample_number']
        self.path = config['path']
        self.oldpath = config['oldpath']
        self.fig_path = config['fig_path']
        self.p_path = config['p_path']
        self.PT = config['pre_training']
        self.device = device
        self.gain = config['gain']
        self.train_epoch_loss, self.train_batch_loss, self.eval_epoch_loss, self.eval_batch_loss, self.eval_batch_accuracy, self.train_epoch_accuracy = [], [], [], [], [], []
        self.train_epoch_roc, self.eval_epoch_roc = [], []
        self.eval_epoch_CE_loss, self.eval_epoch_E_loss, self.eval_epoch_lstm_loss = [], [], []
        self.test_accuracy = 0.
        self.test_auc = 0.
        self.test_loss = 0.
        self.trials = trial
        self.gtrial = gtrial
        self.exp = config['exp']
        self.cv = crossv
        self.attn = nn.Sequential(
            # nn.Linear(2 * self.lstm.hidden_dim, 128),
            nn.Linear(self.lstm.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, 200),
            nn.ReLU(),
            nn.Linear(200,2)

        ).to(device)
        self.init_weight()
        self.dropout = nn.Dropout(0.65).to(self.device)

        if self.exp in ['UFPT', 'NPT']:
            self.optimizer = torch.optim.Adam(
                list(self.decoder.parameters()) + list(self.attn.parameters()) + list(self.lstm.parameters()) + list(
                    self.encoder.parameters()), lr=config['lr'], eps=1e-5)
        else:
            if self.PT in ['milc', 'variable-attention', 'two-loss-milc']:
                self.optimizer = torch.optim.Adam(list(self.decoder.parameters()),lr=config['lr'], eps=1e-5)
            else:
                self.optimizer = torch.optim.Adam(list(self.decoder.parameters() + list(self.attn.parameters())
                                                       + list(self.lstm.parameters())), lr=config['lr'], eps=1e-5)

        self.encoder_backup = self.encoder
        self.lstm_backup = self.lstm
        self.early_stopper = EarlyStopping(self.encoder_backup, self.lstm_backup, patience=self.patience, verbose=False,
                                           wandb=self.wandb, name="encoder",
                                           path=self.path, trial=self.trials)
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def generate_batch(self, episodes, mode):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number
        # print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        if mode == 'train':
            BS = 32
        else:
            BS = len(episodes)
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=False),
                               BS, drop_last=False)

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            ts_number = torch.LongTensor(indices)
            i = 0
            sx = []
            for episode in episodes_batch:
                # Get all samples from this episode
                sx.append(episode)
            yield torch.stack(sx).to(self.device), ts_number.to(self.device)

    def get_saliency_maps(self, X, y, subjects_per_group, trial_no):

        # for param in self.encoder.parameters():
        #     print(param)
        #     param.requires_grad = False
        #
        # for param in self.lstm.parameters():
        #     param.requires_grad = False
        #
        # for param in self.attn.parameters():
        #     param.requires_grad = False
        #
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        print('Computation is running')
        saliencies = []
        counter = 0
        for x, l in zip(X, y):
            counter = counter + 1
            if counter % 100 == 0:
                print('Completed {} samples'.format(counter))

            x.requires_grad_()
            input = self.encoder(x, fmaps=False)
            input = input.unsqueeze(0).to(self.device)

            outputs = self.lstm(input)

            weights_list = []

            for Z in outputs:
                # result = [self.attn(torch.cat((Z[i], Z[-1]), 0)) for i in range(Z.shape[0])]
                result = [self.attn(Z[i]) for i in range(Z.shape[0])]
                result_tensor = torch.stack(result).to(self.device)
                weights_list.append(result_tensor)

            weights = torch.stack(weights_list).to(self.device)
            weights = weights.squeeze(2).to(self.device)

            # SoftMax normalization on weights
            normalized_weights = F.softmax(weights, dim=1)

            # Batch-wise multiplication of weights and lstm outputs

            attn_applied = torch.bmm(normalized_weights.unsqueeze(1).to('cpu'), outputs.to('cpu'))
            attn_applied = attn_applied.squeeze().to(self.device)

            # Pass the weighted output to decoder
            output = self.decoder(attn_applied)
            grad_outputs = torch.zeros(1, 2).to(self.device)

            label = l.long()
            grad_outputs[0][label] = 1

            self.optimizer.zero_grad()

            x.cpu()
            output.backward(gradient=grad_outputs)

            grads = x.grad.data

            saliency = np.squeeze(grads.cpu().numpy())

            saliencies.append(saliency)

        saliencies = np.stack(saliencies, axis=0)
        path = os.path.join(self.path, 'Saliency')
        filename = os.path.join(path, self.exp + '_subj_' + str(subjects_per_group) + '_trial_' + str(trial_no))
        np.save(filename, saliencies)

        return saliencies


    def do_one_epoch(self, epoch, episodes, mode, subjects_per_group, trial_no):
        # mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps, epoch_acc, epoch_roc = 0., 0., 0, 0., 0.
        epoch_CE_loss, epoch_E_loss, epoch_lstm_loss = 0., 0., 0.,
        accuracy1, accuracy2, accuracy, FP = 0., 0., 0., 0.
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_FP = 0., 0., 0., 0.

        data_generator = self.generate_batch(episodes, mode)
        for sx, ts_number in data_generator:
            loss = 0.
            CE_loss, E_loss, lstm_loss = 0., 0., 0.

            # print('sx', sx.device, sx.dtype, type(sx), sx.type())
            inputs = [self.encoder(x, fmaps=False) for x in sx]
            outputs = self.lstm(inputs, mode)

            logits = self.get_attention(outputs)
            logits = logits.to(self.device)

            if mode == 'train':
                targets = self.tr_labels[ts_number]

            elif mode == 'eval':
                targets = self.val_labels[ts_number]

            elif mode == 'test':
                targets = self.test_labels[ts_number]

            targets = targets.long()
            loss = F.cross_entropy(logits, targets.to(self.device))

            # regularization
            if mode == 'train' or mode == 'eval':
                loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)

            accuracy, roc, indices = self.acc_and_auc(logits, mode, targets)
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if mode == 'test':
                test_samples = [episodes[x] for x in range(episodes.shape[0])]
                labels = [self.test_labels[x] for x in range(episodes.shape[0])]
                saliency = self.get_saliency_maps(test_samples, labels, subjects_per_group, trial_no)
                print('Saliency Shape:', saliency.shape)

            epoch_loss += loss.detach().item()
            epoch_accuracy += accuracy.detach().item()

            # if mode == 'train' or mode == 'eval':
                # epoch_CE_loss += CE_loss.detach().item()
                # epoch_E_loss += E_loss
                # epoch_lstm_loss += lstm_loss.detach().item()

            if mode == 'eval' or mode == 'test':
                epoch_roc += roc

            steps += 1

        if mode == "eval":
            self.eval_batch_accuracy.append(epoch_accuracy / steps)
            self.eval_epoch_loss.append(epoch_loss / steps)
            self.eval_epoch_roc.append(epoch_roc / steps)
            self.eval_epoch_CE_loss.append(epoch_CE_loss / steps)
            self.eval_epoch_E_loss.append(epoch_E_loss / steps)
            self.eval_epoch_lstm_loss.append(epoch_lstm_loss / steps)
        elif mode == 'train':
            self.train_epoch_loss.append(epoch_loss / steps)
            self.train_epoch_accuracy.append(epoch_accuracy / steps)
        if epoch % 1 == 0:
          self.log_results(epoch, epoch_loss1 / steps, epoch_loss / steps, epoch_accuracy / steps,
                       epoch_FP / steps, epoch_roc / steps, prefix=mode)
        if mode == "eval":
            self.early_stopper(epoch_loss / steps, epoch_roc / steps, self.encoder, self.lstm, self.attn, self.decoder, 0)
        if mode == 'test':
            self.test_accuracy = epoch_accuracy / steps
            self.test_auc = epoch_roc / steps
            self.test_loss = epoch_loss / steps

        return epoch_loss / steps

    def acc_and_auc(self, logits, mode, targets):
        # N = logits.size(0)
        # sig = torch.zeros(N, 2).to(self.device)
        sig = torch.softmax(logits, dim=1).to(self.device)
        values, indices = sig.max(1)
        roc = 0.
        acc = 0.
        # y_scores = sig.detach().gather(1, targets.to(self.device).long().view(-1,1))
        if mode == 'eval' or mode == 'test':
            y_scores = sig.to(self.device).detach()[:, 1]
            roc = roc_auc_score(targets.to('cpu'), y_scores.to('cpu'))
        accuracy = calculate_accuracy_by_labels(indices, targets.to(self.device))

        return accuracy, roc, indices

    def get_attention(self, outputs):
        # print('Outputs From LSTM:', outputs.shape)

        weights_list = []

        for X in outputs:
            # result = [self.attn(torch.cat((X[i], X[-1]), 0)) for i in range(X.shape[0])]
            result = [self.attn(X[i]) for i in range(X.shape[0])]
            result_tensor = torch.stack(result).to(self.device)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list).to(self.device)
        weights = weights.squeeze().to(self.device)

        # print('Weights Shape:', weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1).to(self.device), outputs.to(self.device))
        attn_applied = attn_applied.squeeze().to(self.device)

        #print('After attention shape:', attn_applied.shape)

        # Pass the weighted output to decoder
        logits = self.decoder(attn_applied)
        return logits

    def add_regularization(self, loss):
        reg = 1e-3
        E_loss = 0.
        lstm_loss = 0.
        attn_loss = 0.
        CE_loss = loss
        # for name, param in self.encoder.named_parameters():
        # if 'bias' not in name:
        # E_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.lstm.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        # for name, param in self.attn.named_parameters():
        #     if 'bias' not in name:
        #         attn_loss += (reg * torch.sum(torch.abs(param)))



        loss = loss + E_loss + lstm_loss
        return loss, CE_loss, E_loss, lstm_loss

    def validate(self, val_eps):

        model_dict = torch.load(os.path.join(self.p_path, 'encoder' + self.trials + '.pt'), map_location=self.device)
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        model_dict = torch.load(os.path.join(self.p_path, 'lstm' + self.trials + '.pt'), map_location=self.device)
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        # model_dict = torch.load(os.path.join(self.p_path, 'decoder' + self.trials + '.pt'), map_location=self.device)
        # self.decoder.load_state_dict(model_dict)
        # self.decoder.eval()
        # self.decoder.to(self.device)

        mode = 'eval'
        self.do_one_epoch(0, val_eps, mode)
        return self.test_auc

    def load_model_and_test(self, tst_eps, subjects_per_group, trial_no):

        filename = os.path.join(self.path, 'encoder' + self.trials + '.pt')
        print('Loading models from: {}'.format(filename))

        model_dict = torch.load(os.path.join(self.path, 'encoder' + self.trials + '.pt'), map_location=self.device)
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        print('Loading LSTM...')

        model_dict = torch.load(os.path.join(self.path, 'lstm' +  self.trials + '.pt'), map_location=self.device)
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        print('Loading Attention')

        model_dict = torch.load(os.path.join(self.path, 'attn' +  self.trials + '.pt'), map_location=self.device)
        self.attn.load_state_dict(model_dict)
        self.attn.eval()
        self.attn.to(self.device)

        print('Loading Decoder')
        model_dict = torch.load(os.path.join(self.path, 'cone' + self.trials + '.pt'), map_location=self.device)
        self.decoder.load_state_dict(model_dict)
        self.decoder.eval()
        self.decoder.to(self.device)


        mode = 'test'
        self.do_one_epoch(0, tst_eps, mode, subjects_per_group, trial_no)

        return self.test_accuracy, self.test_auc

    def save_loss_and_auc(self):

        with open(os.path.join(self.path, 'all_data_information' + self.trials + '.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            self.train_epoch_loss.insert(0, 'train_epoch_loss')
            wr.writerow(self.train_epoch_loss)

            self.train_epoch_accuracy.insert(0, 'train_epoch_accuracy')
            wr.writerow(self.train_epoch_accuracy)

            self.eval_epoch_loss.insert(0, 'eval_epoch_loss')
            wr.writerow(self.eval_epoch_loss)

            self.eval_batch_accuracy.insert(0, 'eval_batch_accuracy')
            wr.writerow(self.eval_batch_accuracy)

            self.eval_epoch_roc.insert(0, 'eval_epoch_roc')
            wr.writerow(self.eval_epoch_roc)

            self.eval_epoch_CE_loss.insert(0, 'eval_epoch_CE_loss')
            wr.writerow(self.eval_epoch_CE_loss)

            self.eval_epoch_E_loss.insert(0, 'eval_epoch_E_loss')
            wr.writerow(self.eval_epoch_E_loss)

            self.eval_epoch_lstm_loss.insert(0, 'eval_epoch_lstm_loss')
            wr.writerow(self.eval_epoch_lstm_loss)

    def train(self, tr_eps, val_eps, tst_eps, subjects_per_group, trial_no):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 30, 128, 256, 512, 700, 800, 2500], gamma=0.15)
        e =0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        if self.PT in ['milc', 'variable-attention', 'two-loss-milc']:
            if self.exp in ['UFPT', 'FPT']:
                model_dict = torch.load(os.path.join(self.oldpath, 'lstm' + '.pt'))
                self.lstm.load_state_dict(model_dict)
                self.lstm.to(self.device)

                model_dict = torch.load(os.path.join(self.oldpath, 'attn' + '.pt'))
                self.attn.load_state_dict(model_dict)
                self.attn.to(self.device)
        saved = 0
        for e in range(self.epochs):
            if self.exp in ['UFPT', 'NPT']:
                self.encoder.train(), self.lstm.train(), self.attn.train()
            else:
                self.encoder.eval(), self.lstm.train(), self.attn.train()

            mode = "train"
            val_loss = self.do_one_epoch(e, tr_eps, mode, subjects_per_group, trial_no)

            self.encoder.eval(), self.lstm.eval(), self.attn.eval()
            mode = "eval"
            val_loss = self.do_one_epoch(e, val_eps, mode, subjects_per_group, trial_no)
            scheduler.step(val_loss)
            if self.early_stopper.early_stop:
                self.early_stopper(0, 0, self.encoder, self.lstm, self.attn, self.decoder, 1)
                saved = 1
                break

        if saved == 0:
            self.early_stopper(0, 0, self.encoder, self.lstm, self.attn, self.decoder, 1)
            saved = 1

        self.save_loss_and_auc()
        self.load_model_and_test(tst_eps, subjects_per_group, trial_no)


        return self.test_accuracy, self.test_auc, self.test_loss, e
        # return self.early_stopper.val_acc_max



    def log_results(self, epoch_idx, epoch_loss1, epoch_loss, epoch_test_accuracy, epoch_FP, epoch_roc, prefix=""):
        print(
            "{} CV: {}, Trial: {}, Epoch: {}, Epoch Loss: {}, Epoch Accuracy: {}, Epoch FP: {} roc: {},  {}".format(
                prefix.capitalize(),
                self.cv,
                self.trials,
                epoch_idx,
                epoch_loss,
                epoch_test_accuracy,
                epoch_FP,
                epoch_roc,
                prefix.capitalize()))
