import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch.utils.data import RandomSampler, BatchSampler
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from source.utils import get_argparser
from scripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2


parser = get_argparser()
args = parser.parse_args()


components = 50
time_points = 140
window_shift = 20
samples_per_subject = 7
sample_y = 20


def ReLU(x):
    return x * (x > 0)

Data, labels, start_positions, masks = artificial_batching_patterned_space2(20000, 140, 50, seed=1988)
print(Data[0, 0, 0:20])
Data = np.moveaxis(Data, 1, 2)

print('Original Data Shape:', Data.shape)


FinalData = Data[18000:]
Labels = labels[18000:]
print('Test Data Shape:', labels.shape)

X = np.zeros((2000, components, samples_per_subject * sample_y))

# k = 100
# Read Saliency
filename = "../wandb"
filename = os.path.join(filename, 'Sequence_Based_Models')
filename = os.path.join(filename, 'MILC_100_models_saliencies', 'new_dir_avg_sal_raw.npy')
saliency = np.load(filename)
# print('Saliency {} loaded successfully...'.format(k))

A = saliency
print(saliency.shape)

# Step - 1: Take ReLU
B = ReLU(A)
saliency = np.abs(saliency)

# for i in range(A.shape[0]):
#     # saliency[i, :, :, :] = saliency[i, :, :, :] / np.amax(saliency[i, :, :, :])
#     saliency[i] = saliency[i] / np.max(saliency[i])


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

# D = np.random.choice(5000, 10, replace=False)
D = np.asarray(range(20, 30, 1))
print(D)
L = Labels[D[:]]
print('Labels:', L)


fig, axes = plt.subplots(10, 2)
for i in range(10):
    axes[i, 0].imshow(FinalData[D[i]], interpolation='nearest', aspect='auto', cmap='Reds')
    axes[i, 1].imshow(avg_saliency[D[i]], interpolation='nearest', aspect='auto', cmap='Reds')

# Turn off *all* ticks & spines, not just the ones with colormaps.

for i in range(10):
    axes[i, 0].set_axis_off()
    axes[i, 1].set_axis_off()

axes[0, 0].set_title('Original Data')
axes[0, 1].set_title('Saliency')

plt.show()

path = "../wandb"
path = os.path.join(path, 'Sequence_Based_Models', 'MILC_100_models_saliencies')
path = os.path.join(path, '', 'newDirAvgMap' + '.png')
fig.savefig(path, format='png', dpi=600)



