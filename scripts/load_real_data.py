'''
This Data Loader is written for loading real ICA data to be used in the server in parallel
'''

import numpy as np
import torch

import pandas as pd
import h5py
import time
import os

def LoadAddiction():
    sample_x = 100
    subjects = 706
    no_good_comp = 53
    tc = 143
    
    # Read file
    hf = h5py.File('../Addiction_Data/RevisedAddictionAll.h5', 'r')
    AddictionData = hf.get('Addiction_Data')
    AddictionData = np.array(AddictionData)
    print('Fire read correctly..')
    print(AddictionData.shape)
    hf.close()

    filename = '../Addiction_Data/correct_indices_ADC.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData = AddictionData[:, c_indices, 0: 140]
    print('Reshape we finalized:', finalData.shape)

    FNC = np.zeros((subjects, 1378))
    corrM = np.zeros((subjects, no_good_comp, no_good_comp))
    for i in range(subjects):
        corrM[i, :, :] = np.corrcoef(finalData[i])
        M = corrM[i, :, :]
        FNC[i, :] = M[np.triu_indices(53, k=1)]

    print(FNC.shape)

    filename = '../Addiction_Data/index_array_for_shuffle_addiction.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    filename = '../Addiction_Data/revised_criteria_2_labels_addiction.csv'
    labels = pd.read_csv(filename, header=None)
    print(labels)
    
    all_labels = labels.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    FNC = FNC[index_array, :]
    all_labels = all_labels[index_array.long()]
    finalData =finalData[index_array, :, :]
    return  finalData, FNC, all_labels

def LoadFBIRN():
    sample_x = 100
    subjects = 311
    subjects_for_test = 64
    subjects_for_val = 47
    no_good_comp = 53
    tc = 140

    hf = h5py.File('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/FBIRN/FBIRN_AllData.h5', 'r')
    data2 = hf.get('FBIRN_dataset')
    data2 = np.array(data2)
    print(data2.shape)
    data2 = data2.reshape((subjects, sample_x, tc))
    data = data2
    print('Reshape we need:', data.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/FBIRN/correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]
    print('Reshape we finalized:', finalData.shape)

    FNC = np.zeros((subjects, 1378))
    corrM = np.zeros((subjects, no_good_comp, no_good_comp))
    for i in range(subjects):
        corrM[i, :, :] = np.corrcoef(finalData[i])
        M = corrM[i, :, :]
        FNC[i, :] = M[np.triu_indices(53, k=1)]

    print(FNC.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/FBIRN/index_array_labelled_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/FBIRN/labels_FBIRN.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    FNC = FNC[index_array, :]
    all_labels = all_labels[index_array.long()]
    finalData =finalData[index_array, :, :]
    return  finalData, FNC, all_labels

def LoadCOBRE():
    sample_x = 100
    subjects = 157
    subjects_for_test = 32
    subjects_for_val = 32
    no_good_comp = 53
    tc = 140

    ntrain_samples = 93
    ntest_samples = 32

    hf = h5py.File('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/COBRE/COBRE_AllData.h5', 'r')
    data2 = hf.get('COBRE_dataset')
    data2 = np.array(data2)
    print(data2.shape)
    data2 = data2.reshape((subjects, sample_x, tc))
    data = data2
    print('Reshape we need:', data.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/COBRE/correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]
    print('Reshape we finalized:', finalData.shape)

    FNC = np.zeros((subjects, 1378))
    corrM = np.zeros((subjects, no_good_comp, no_good_comp))
    for i in range(subjects):
        corrM[i, :, :] = np.corrcoef(finalData[i])
        M = corrM[i, :, :]
        FNC[i, :] = M[np.triu_indices(53, k=1)]

    print(FNC.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/COBRE/index_array_labelled_COBRE2.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/COBRE/labels_COBRE.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1

    FNC = FNC[index_array, :]
    all_labels = all_labels[index_array.long()]
    finalData=finalData[index_array, :, :]
    return  finalData, FNC, all_labels


def LoadOASIS():
    sample_x = 100
    subjects = 372
    subjects_for_test = 64
    subjects_for_val = 64
    no_good_comp = 53
    tc = 120

    ntrain_samples = 244
    ntest_samples = 64

    hf = h5py.File('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/OASIS/OASIS3_AllData.h5', 'r')
    data2 = hf.get('OASIS3_dataset')
    data2 = np.array(data2)
    print(data2.shape)
    data2 = data2.reshape((subjects, sample_x, tc))
    data = data2
    print('Reshape we need:', data.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/OASIS/correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]
    print('Reshape we finalized:', finalData.shape)

    FNC = np.zeros((subjects, 1378))
    corrM = np.zeros((subjects, no_good_comp, no_good_comp))
    for i in range(subjects):
        corrM[i, :, :] = np.corrcoef(finalData[i])
        M = corrM[i, :, :]
        FNC[i, :] = M[np.triu_indices(53, k=1)]

    print(FNC.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/OASIS/index_array_labelled_OASIS3.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/OASIS/labels_OASIS3.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)

    # all_labels = all_labels - 1

    FNC = FNC[index_array, :]
    all_labels = all_labels[index_array.long()]
    finalData=finalData[index_array, :, :]
    return  finalData, FNC, all_labels

def LoadABIDE():

    print('Welcome to ABIDE Analysis')
    sample_x = 100
    subjects = 569
    subjects_for_test = 64
    subjects_for_val = 47
    no_good_comp = 53
    tc = 140

    ntrain_samples = 469
    ntest_samples = 100

    hf = h5py.File('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/ABIDE/ABIDE1_AllData.h5', 'r')
    data2 = hf.get('ABIDE1_dataset')
    data2 = np.array(data2)
    print(data2.shape)
    data2 = data2.reshape((subjects, sample_x, tc))
    data = data2
    print('Reshape we need:', data.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/ABIDE/correct_indices_GSP.csv'
    print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]
    print('Reshape we finalized:', finalData.shape)

    FNC = np.zeros((subjects, 1378))
    corrM = np.zeros((subjects, no_good_comp, no_good_comp))
    for i in range(subjects):
        corrM[i, :, :] = np.corrcoef(finalData[i])
        M = corrM[i, :, :]
        FNC[i, :] = M[np.triu_indices(53, k=1)]

    print(FNC.shape)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/ABIDE/index_array_labelled_ABIDE1.csv'   # Permutation
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(subjects)

    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/ABIDE/labels_ABIDE1.csv'   # Main order before permutation
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
#     print(all_labels)
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    FNC = FNC[index_array, :]
    all_labels = all_labels[index_array.long()]
    finalData=finalData[index_array, :, :]
    return finalData, FNC, all_labels

def LoadHCP():
    
    "It returns HCP data formatted with stride = 1 window shift."
    
    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/index_array_HCP.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(823)

    if not os.path.exists('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPData.npz'):
        # For ICA TC Data
        data = np.zeros((823, 100, 1040))
        # modifieddata = np.zeros((823, 100, 1100))
        finalData = np.zeros((823, 1020, 100, 20))
        finalData2 = np.zeros((823, 1020, 53, 20))
        for p in range(823):
            # print(p)
            filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/TC/HCP_ica_br' + str(p + 1) + '.csv'
            # filename = '../TimeSeries/HCP_ica_br1.csv'
            if p % 20 == 0:
                print(filename)
            df = pd.read_csv(filename, header=None)
            d = df.values
            data[p, :, :] = d[:, 0:1040]

        for i in range(823):
            for j in range(1020):
                finalData[i, j, :, :] = data[i, :, j * 1:j * 1 + 20]

        print(finalData.shape)

        filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/correct_indices_HCP.csv'
        print(filename)
        df = pd.read_csv(filename, header=None)
        c_indices = df.values
        c_indices = torch.from_numpy(c_indices).int()
        c_indices = c_indices.view(53)
        c_indices = c_indices - 1
        finalData2 = finalData[:, :, c_indices, :]
        print(c_indices)
        print(finalData2.shape)
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPData.npz', 'wb') as file:
            np.save(file, finalData2)
            print('Data Saved Successfully')

    else:
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPData.npz', 'rb') as file:
            finalData2 = np.load(file)
            print(finalData2.shape)
            print('Data loaded successfully...')
    return finalData2, index_array



def LoadHCPStride10():
    
    "It returns HCP data formatted with stride = 10 window shift."
    
    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/index_array_HCP.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(823)

    if not os.path.exists('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPDataStride10.npz'):
        # For ICA TC Data
        data = np.zeros((823, 100, 1040))
        # modifieddata = np.zeros((823, 100, 1100))
        finalData = np.zeros((823, 103, 100, 20))
        finalData2 = np.zeros((823, 103, 53, 20))
        for p in range(823):
            # print(p)
            filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/TC/HCP_ica_br' + str(p + 1) + '.csv'
            # filename = '../TimeSeries/HCP_ica_br1.csv'
            if p % 20 == 0:
                print(filename)
            df = pd.read_csv(filename, header=None)
            d = df.values
            data[p, :, :] = d[:, 0:1040]

        for i in range(823):
            for j in range(103):
                finalData[i, j, :, :] = data[i, :, j * 10:j * 10 + 20]

        print(finalData.shape)

        filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/correct_indices_HCP.csv'
        print(filename)
        df = pd.read_csv(filename, header=None)
        c_indices = df.values
        c_indices = torch.from_numpy(c_indices).int()
        c_indices = c_indices.view(53)
        c_indices = c_indices - 1
        finalData2 = finalData[:, :, c_indices, :]
        print(c_indices)
        print(finalData2.shape)
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPDataStride10.npz', 'wb') as file:
            np.save(file, finalData2)
            print('Data Saved Successfully')

    else:
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPDataStride10.npz', 'rb') as file:
            finalData2 = np.load(file)
            print(finalData2.shape)
            print('Data loaded successfully...')
    return finalData2, index_array

def LoadHCPStride20():
    
    "It returns HCP data formatted with stride = 20 window shift."
    
    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/index_array_HCP.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(823)

    if not os.path.exists('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPDataStride20.npz'):
        # For ICA TC Data
        data = np.zeros((823, 100, 1040))
        # modifieddata = np.zeros((823, 100, 1100))
        finalData = np.zeros((823, 52, 100, 20))
        finalData2 = np.zeros((823, 52, 53, 20))
        for p in range(823):
            # print(p)
            filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/TC/HCP_ica_br' + str(p + 1) + '.csv'
            # filename = '../TimeSeries/HCP_ica_br1.csv'
            if p % 20 == 0:
                print(filename)
            df = pd.read_csv(filename, header=None)
            d = df.values
            data[p, :, :] = d[:, 0:1040]

        for i in range(823):
            for j in range(52):
                finalData[i, j, :, :] = data[i, :, j * 20:j * 20 + 20]

        print(finalData.shape)

        filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/correct_indices_HCP.csv'
        print(filename)
        df = pd.read_csv(filename, header=None)
        c_indices = df.values
        c_indices = torch.from_numpy(c_indices).int()
        c_indices = c_indices.view(53)
        c_indices = c_indices - 1
        finalData2 = finalData[:, :, c_indices, :]
        print(c_indices)
        print(finalData2.shape)
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPDataStride20.npz', 'wb') as file:
            np.save(file, finalData2)
            print('Data Saved Successfully')

    else:
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPDataStride20.npz', 'rb') as file:
            finalData2 = np.load(file)
            print(finalData2.shape)
            print('Data loaded successfully...')
    return finalData2, index_array



def LoadHCPFull():
    filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/index_array_HCP.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).int()
    index_array = index_array.view(823)

    if not os.path.exists('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPFull.npz'):
        # For ICA TC Data
        data = np.zeros((823, 100, 1040))
        # modifieddata = np.zeros((823, 100, 1100))
#         finalData = np.zeros((823, 1020, 100, 20))
#         finalData2 = np.zeros((823, 1020, 53, 20))
        for p in range(823):
            # print(p)
            filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/TC/HCP_ica_br' + str(p + 1) + '.csv'
           
            if p % 20 == 0:
                print(filename)
            df = pd.read_csv(filename, header=None)
            d = df.values
            print(d.shape)
            data[p, :, :] = d[:, 0:1040]

   
        filename = '/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCP/correct_indices_HCP.csv'
        print(filename)
        df = pd.read_csv(filename, header=None)
        c_indices = df.values
        c_indices = torch.from_numpy(c_indices).int()
        c_indices = c_indices.view(53)
        c_indices = c_indices - 1
        finalData2 = data[:, c_indices, :]
        print(c_indices)
        print(finalData2.shape)
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPFull.npz', 'wb') as file:
            np.save(file, finalData2)
            print('Data Saved Successfully')

    else:
        with open('/data/users2/mrahman21/My_Project/baselines/pytorch-a2c-ppo-acktr-gail/STDIM_fMRI/HCPFull.npz', 'rb') as file:
            finalData2 = np.load(file)
            print(finalData2.shape)
            print('Data loaded successfully...')
    return finalData2, index_array
