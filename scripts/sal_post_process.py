import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scripts.load_FBIRN_Data import load_FBIRN_DATA
from scipy import stats
import torch
from torch.utils.data import RandomSampler, BatchSampler
from scripts.LoadRealData import LoadABIDE, LoadCOBRE, LoadFBIRN, LoadOASIS
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from src.usman_utils import get_argparser
from scripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2

'''

Version - 2: 

This code is written for saliency post-processing:
    --extraction
    --filtering
    --marker detection
    --FNC computation from markers 

Steps are:
    1) Take ReLU
    2) Normalize subjectwise with Max
    3) Take average on each time point
    4) Use Butterworth filter with order = 6 , bandwidth = 0.2
    5) Use filtfilt
    6) Use some thresholding 
'''


parser = get_argparser()
args = parser.parse_args()


components = 50
time_points = 140
window_shift = 20
samples_per_subject = 7
sample_y = 20


def ReLU(x):
    return x * (x > 0)


# def find_indices_of_each_class(all_labels):
#     HC_index = (all_labels == 0).nonzero()
#     SZ_index = (all_labels == 1).nonzero()
#
#     return HC_index, SZ_index


# Load labels according to the saliency order

# filename = '../OASIS/index_array_labelled_OASIS3.csv'
# df = pd.read_csv(filename, header=None)
# index_array = df.values
# index_array = torch.from_numpy(index_array).long()
# index_array = index_array.view(index_array.size(0))
#
# filename = '../OASIS/labels_OASIS3.csv'
# df = pd.read_csv(filename, header=None)
# all_labels = df.values
# all_labels = torch.from_numpy(all_labels).int()
# all_labels = all_labels.view(all_labels.size(0))
# # all_labels = all_labels - 1
# all_labels = all_labels[index_array]
#
# HC_index, SZ_index = find_indices_of_each_class(all_labels)
#
# # Load Test Labels
# cls = ['HC', 'SZ']

#
# # Load the Noah's Syn Data (Data NOT Saliency)
# All = np.load('../Noah Synthetic Data/Data.npy')
# labels = np.load('../Noah Synthetic Data/Labels.npy')
# # np.load('../Noah Synthetic Data/start_positions.npy')
# np.load('../Noah Synthetic Data/masks.npy')
# print('Data Loaded Successfully')
# print('Original Data Shape:', All.shape)


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
    print('Model without flip')

FinalData = Data[27500:, :, :]
Labels = labels[27500:]

print(Data.shape)

# Read Saliency

filename = os.path.join(os.getcwd(), 'wandb', 'Sequence_Based_Models')
# filename = os.path.join(filename, 'OASIS_str_1', 'Saliency', 'UFPT_subj_0_trial_4.npz')
# filename = os.path.join(filename, 'OASIS_HOL_LE', 'Saliency', 'UFPT_subj_0_trial_8.npz')
# filename = os.path.join(filename, 'OASIS_str_1_both_PTNT', 'Saliency', 'UFPT_subj_0_trial_5_predicted.npz')
filename = os.path.join(filename, 'New_Syn_Data_No_Overlap_Flipped_1', 'Saliency', 'NPT_subj_0_trial_0.npy')


with open(filename, 'rb') as file:
    saliency = np.load(file)
    print('Saliency loaded successfully...')

A = saliency
print(saliency.shape)

# Step - 1: Take ReLU
B = ReLU(A)
C = ReLU(-A)
saliency = np.abs(saliency)


# Step - 2: Normalize Subjectwise with Max
# for i in range(A.shape[0]):
    # B[i, :, :, :] = B[i, :, :, :] / np.amax(B[i, :, :, :])
    # B[i, :, :, :] = B[i, :, :, :] - np.mean(B[i, :, :, :])
    # B[i, :, :, :] = B[i, :, :, :] / np.std(B[i, :, :, :])
    # C[i, :, :, :] = C[i, :, :, :] - np.mean(C[i, :, :, :])
    # C[i, :, :, :] = C[i, :, :, :] / np.std(C[i, :, :, :])

# print(B)
saliency = B #C # B+C
# print(C)
# print(A)

# for i in range(A.shape[0]):
#     saliency[i, :, :, :] = saliency[i, :, :, :] / np.amax(saliency[i, :, :, :])


saliency = np.moveaxis(saliency, 1, 2)
# Stitch all the frames together
# stiched_saliency = np.zeros((A.shape[0], components, samples_per_subject * sample_y))
# for i in range(A.shape[0]):
#     for j in range(A.shape[1]):
#         stiched_saliency[i, :, j * 20:j * 20 + sample_y] = saliency[i, j, :, :]
#
# saliency = stiched_saliency

# For NO Overlapping # str = 20
# ================== #

avg_saliency = saliency

print('Avg Shape:', avg_saliency.shape)


# avg_saliency = np.zeros((A.shape[0], components, time_points))
#
# # For Old Overlapping # str = 10
# # =================== #
#
# avg_saliency[:, :, 0:10] = saliency[:, :, 0:10]
#
# for j in range(samples_per_subject-1):
#     a = saliency[:, :, 20*j+10:20*j+20]
#     # print('A Shape:', a.shape)
#     b = saliency[:, :, 20*(j+1):20*(j+1)+10]
#     # print('B Shape:', b.shape)
#     avg_saliency[:, :, 10*j+10:10*j+20] = (a + b)/2
#
# avg_saliency[:, :, time_points-10:time_points] = saliency[:, :, samples_per_subject*sample_y-10:samples_per_subject*sample_y]
# print('Avg Saliency Shape:', avg_saliency.shape)


# For New Overlapping str = 1
# ===================

# for i in range(A.shape[0]):
#     for j in range(time_points):
#         L = []
#         if j < 20:
#             index = j
#         else:
#             index = 19 + (j - 19) * 20
#
#         L.append(index)
#
#         s = saliency[i, :, index]
#         count = 1
#         block = 1
#         iteration = min(19, j)
#         for k in range(0, iteration, 1):
#             if index + block * 20 - (k + 1) < samples_per_subject * sample_y:
#                 s = s + saliency[i, :, index + block * 20 - (k + 1)]
#                 L.append(index + block * 20 - (k + 1))
#                 count = count + 1
#                 block = block + 1
#             else:
#                 break
#         avg_saliency[i, :, j] = s / count
#         # print('Count =', count, ' and Indices =', L)
#
# print('Average Saliency Shape:', avg_saliency.shape)

# # Middle time point idea - take middle time step from each window
# new_saliency = np.zeros((saliency.shape[0], components, samples_per_subject))
#
# for i in range(saliency.shape[0]):
#     for j in range(samples_per_subject):
#         new_saliency[i, :, j] = saliency[i, :, 9+20*j]
#
# print("New Shape:", new_saliency.shape)

# Step - 3: Take Average on Each Time point

# comp_avged_saliency = np.zeros((avg_saliency.shape[0], time_points))
#
# for i in range(avg_saliency.shape[0]):
#     T = avg_saliency[i, :, :]
#     B = np.mean(T, axis=0)
#     comp_avged_saliency[i, :] = B

# Step - 4: Use Butterworth Filter with order  = 6, bandwidth = 0.2

# b, a = signal.butter(6, 0.2, 'low', analog=False, output='ba')

# Step - 5: Use filtering parameter obtained from step - 4

# filtered_avg_sal = signal.filtfilt(b, a, comp_avged_saliency)
# print('Output Shape = {}'.format(filtered_avg_sal.shape))

# Visualize Saliency

# smoothen the maps
# 1-D butterworth

# for i in range(avg_saliency.shape[0]):
#     for j in range(avg_saliency.shape[1]):
#         new_saliency[i, j, : ]=signal.filtfilt(b, a, new_saliency[i, j, :])
#         new_saliency[i, j, :] = gaussian_filter1d(new_saliency[i, j, :], sigma=1)

# gaussian filter

# for i in range(avg_saliency.shape[0]):
#     new_saliency[i, :, :] = gaussian_filter(new_saliency[i, :, :], sigma=1)

# HC = np.random.choice(HC_index.shape[0], 10, replace=False)
# SZ = np.random.choice(SZ_index.shape[0], 10, replace=False)

# HC = [177, 154,  48,  54,  61, 163, 148,  72,  85,  47]
# SZ = [170,  25, 168,  22,  19, 5, 152, 133, 151, 172]

# HC = [ 82,  27,   9,  60, 148,   8, 128, 173,   6,  47]
# SZ = [ 44, 144,  54, 181,  57, 133, 153,  65, 171, 183]


# print(HC)
# print(SZ)
# D = np.random.choice(5000, 10, replace=False)
D = np.asarray(range(20, 30, 1))
print(D)
L = Labels[D[:]]
print('Labels:', L)

# choice = np.random.choice(A.shape[0], 1, replace=False)
# print('choice: {}'.format(choice))

fig, axes = plt.subplots(10, 2)
for i in range(10):
    # plt.subplot(10, 2, 2*i+1)
    # axes[i, 0].imshow(avg_saliency[HC_index[HC[i]]], interpolation='nearest', aspect='auto', cmap='Reds')
    # axes[i, 1].imshow(avg_saliency[SZ_index[SZ[i]]], interpolation='nearest', aspect='auto', cmap='Reds')
    axes[i, 0].imshow(Data[D[i]], interpolation='nearest', aspect='auto', cmap='Reds')
    # axes[i, 1].imshow(stiched_saliency[D[i]], interpolation='nearest', aspect='auto', cmap='Reds')
    axes[i, 1].imshow(avg_saliency[D[i]], interpolation='nearest', aspect='auto', cmap='Reds')

# Turn off *all* ticks & spines, not just the ones with colormaps.

for i in range(10):
    axes[i, 0].set_axis_off()
    axes[i, 1].set_axis_off()
    # axes[i, 2].set_axis_off()

axes[0, 0].set_title('Original Data')
axes[0, 1].set_title('Saliency')
# axes[0, 2].set_title('Avgd. Saliency')


plt.show()

path = os.path.join(os.getcwd(), 'wandb', 'Sequence_Based_Models')
path = os.path.join(path, 'New_Syn_Data_No_Overlap_Flipped_1', 'flipped_1_lstm_input_relu_' + str(1) + '.png')
fig.savefig(path, format='png', dpi=600)


# plt.subplot(1, 2, 1)
# plt.plot(comp_avged_saliency[choice[0]])
# plt.xlabel('Time Points')
# plt.ylabel('Averaged Over Components')
# plt.title('Before Filtering')
# plt.subplot(1, 2, 2)
# plt.plot(filtered_avg_sal[choice[0]])
# plt.xlabel('Time Points')
# plt.ylabel('Averaged Over Components')
# plt.title('After Filtering')

# plt.show()

# Step 6 : Thresholding - Find the time point that gives maximum average change

# idx = {}
# for i in range(saliency.shape[0]):
#     idx[i] = np.where(filtered_avg_sal[i] == np.amax(filtered_avg_sal[i]))
#
# sal_index = np.zeros(saliency.shape[0])
# for k, v in idx.items():
#     print(str(k) + "\t" + str(v[0][0]))
#     sal_index[int(k)] = v[0][0]
#     # print(v)
#
# # At which time point, for each subject, we are getting salient point
#
# filename = './wandb/Sequence_Based_Models/OASIS_str_1/Salient_Point_Indices.csv'
# np.savetxt(filename, sal_index, fmt='%i', delimiter=',')
# print(' Indices saved here...', filename)
