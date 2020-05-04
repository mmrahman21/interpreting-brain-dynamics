'''
This code is intended for saliency computation on Noah's Synthetic Data
Using MILC
'''

import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import sys
import os

from source.utils import get_argparser
from source.encoders_ICA import NatureCNN, ImpalaCNN, NatureOneCNN, LinearEncoder


for p in sys.path:
    print(p)
from source.MILC_Trainer import LSTMTrainer
from source.ATTLSTM import subjLSTM
import wandb
import pandas as pd

import matplotlib.pyplot as plt
from myscripts.createDirectories import  create_Directories
from myscripts.save_metrics import save_metrics
from numpy import savetxt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from scripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2



def train_encoder(args):
    start_time = time.time()

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + str(cudaID))
    else:
        device = torch.device("cpu")
    print('device = ', device)

    # args.script_ID = 3
    JOB = 3
    print('Script ID:', args.script_ID)
    run_dir = create_Directories(args, 'sequence')

    # dir = "MyRunDir"
    # wdb = 'wandb'
    # wpath = os.path.join(os.getcwd(), wdb)
    # sbpath = os.path.join(wpath, 'Sequence_Based_Models')
    # path = os.path.join(sbpath, dir)
    # run_dir = path
    # args.path = path

    print('Directory Created:', run_dir)

    param = {1: 'FPT', 2: 'UFPT', 3: 'NPT'}
    exp_type = param[JOB]
    args.exp = exp_type
    print("Experiment Running:", exp_type)
    print('pretraining = ' + args.pre_training)

    # Milc default
    if args.exp == 'FPT':
        gain = [0.1, 0.5, 0.35, 0.3, 0.7, 0.4]  # FPT
    elif args.exp == 'UFPT':
        gain = [0.05, 0.15, 0.3, 0.35, 0.65, 0.65]  # UFPT
    else:
        gain = [0.25, 0.35, 0.75, 0.30, 0.4, 0.35]  # NPT


    ID = 4
    current_gain = gain[ID]
    args.gain = current_gain

    # Data = np.load('../Noah Synthetic Data/Data.npy')
    # labels = np.load('../Noah Synthetic Data/Labels.npy')
    # np.load('../Noah Synthetic Data/start_positions.npy')
    # np.load('../Noah Synthetic Data/masks.npy')
    # print('Data Loaded Successfully')



    Data, labels, start_positions, masks = artificial_batching_patterned_space2(30000, 140, 50, seed=1988)
    print(Data[0, 0, 0:20])
    Data = np.moveaxis(Data, 1, 2)

    print('Original Data Shape:', Data.shape)
    print('Original Label Shape:', labels.shape)

    new_data = np.zeros((30000, 50, 140))
    new_label = np.zeros(30000)

    if args.script_ID == 2:
        for i in range(30000):
            new_data[i, :, :] = np.flipud(Data[i, :, :])
            # new_label[i] = 1-labels[i]

        print('New Shape:', new_data.shape)
        Data = new_data
        print('Data is flipped')
        # labels = new_label

    else:
        print('Model is running without flipping')

    subjects = 30000
    sample_x = 50
    sample_y = 20
    tc = 140
    samples_per_subject = 7 #7 #121
    window_shift = 20

    finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))

    for i in range(subjects):
        for j in range(samples_per_subject):
            finalData[i, j, :, :] = Data[i, :, (j * window_shift):(j * window_shift) + sample_y]

    print('Data shape ended up with:', finalData.shape)

    for i in range(1):
        tr_eps = finalData[0:25000, :, :, :]
        val_eps = finalData[25000:27500, :, :, :]
        test_eps = finalData[27500:, :, :, :]

        tr_labels = labels[0:25000]
        val_labels = labels[25000:27500]
        test_labels = labels[27500:]



        #================== Prepare for one time test ============#

        # args.path = './wandb/Sequence_Based_Models/OASIS_str_1_both_PTNT/'

        # args.path = '../wandb/Sequence_Based_Models/New_Syn_Data_No_Overlap_Flipped_2/'

        # For model selection

        # test_eps = hold_out
        # test_labels = hold_out_labels
        #
        # # This training and validation are just placeholders, will not be used for one time test
        # tr_eps = hold_out
        # tr_labels = hold_out_labels
        # val_eps = hold_out
        # val_labels = hold_out_labels

        # # For saliency computation

        # test_eps = full_data
        # test_labels = full_labels
        #
        # # This training and validation are just placeholders, will not be used for one time test
        # tr_eps = full_data
        # tr_labels = full_labels
        # val_eps = full_data
        # val_labels = full_labels
        #
        # #
        # tr_eps = torch.from_numpy(tr_eps).float()
        # val_eps = torch.from_numpy(val_eps).float()
        # test_eps = torch.from_numpy(test_eps).float()
        #
        # tr_eps.to(device)
        # val_eps.to(device)
        # test_eps.to(device)

        #=============================================================================#

        tr_eps = torch.from_numpy(tr_eps).float()
        val_eps = torch.from_numpy(val_eps).float()
        test_eps = torch.from_numpy(test_eps).float()

        tr_labels = torch.from_numpy(tr_labels)
        val_labels = torch.from_numpy(val_labels)
        test_labels = torch.from_numpy(test_labels)

        tr_eps.to(device)
        val_eps.to(device)
        test_eps.to(device)

        print("trainershape", tr_eps.shape)
        print("valshape", val_eps.shape)
        print("testshape", test_eps.shape)
        print('ID = ', args.script_ID)


        observation_shape = finalData.shape
        if args.encoder_type == "Nature":
            encoder = NatureCNN(observation_shape[2], args)


        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(observation_shape[2], args)

        elif args.encoder_type == "NatureOne":
            dir = ""
            if args.pre_training == "milc":
                # dir = 'wandb/realData_pretrained_encoder/Milc_rw_bs_32/encoder.pt'
                # args.oldpath = os.path.join(os.getcwd(), 'wandb/realData_pretrained_encoder/Milc')
                dir = '../wandb/realData_pretrained_encoder/Milc_rw_bs_32/encoder.pt'
                args.oldpath = os.path.join('../', 'wandb/realData_pretrained_encoder/Milc_rw_bs_32')

            path = os.path.join(os.getcwd(), dir)
            encoder = NatureOneCNN(observation_shape[2], args)
            lstm_model = subjLSTM(device, args.feature_size, args.lstm_size, num_layers=args.lstm_layers,
                                  freeze_embeddings=True, gain=current_gain)

            model_dict = torch.load(path, map_location=device)  # with good components


            if args.exp in ['UFPT', 'FPT']:
                encoder.load_state_dict(model_dict)
            encoder.to(device)
            lstm_model.to(device)


        config = {}
        config.update(vars(args))

        if args.method == "sub-lstm":
            trainer = LSTMTrainer(encoder, lstm_model, config, device=device, exp=exp_type, tr_labels=tr_labels,
                                  val_labels=val_labels, test_labels=test_labels,
                                  wandb=wandb, trial=str(i))    #fold number to be set during one time test or fold number
        else:
            assert False, "method {} has no trainer".format(args.method)

    #===================For one time test only ==========#
        # acc, auc = trainer.load_model_and_test(test_eps, 0, 0)
        # print('AUC: {} \n Acc: {}'.format(auc, acc))
    #====================================================#

        accuracy, auc, loss, required_epochs = trainer.train(tr_eps, val_eps,test_eps, 0, i )

    # save_metrics(exp_type, accuracy, loss, auc, required_epochs, run_dir)
    elapsed_time = time.time() - start_time
    print('Total Time Elapsed:', elapsed_time)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
