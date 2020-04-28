import numpy as np
import scipy.stats
import glob
import sys
import os
from scripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2

components = 50
time_points = 140
window_shift = 20
samples_per_subject = 7
sample_y = 20

def robyn_method(Mask,Data):
    X = Mask*Data
    Y = Data
    Xs = np.sum(X)
    Ys = np.sum(Y)
    return Xs/Ys

def spear_rank(M, N):
    M = M.flatten()
    N = N.flatten()

    Ma = np.flip(np.argsort(M))
    Na = np.flip(np.argsort(N))

    Mv = np.copy(M)
    Nv = np.copy(N)

    count = M.shape[0]
    diff = 0
    for i in range(int(M.shape[0] / 8)):
        Mv[Ma[i]] = count
        Nv[Na[i]] = count
        count -= 1
        print(str(Ma[i]) + " " + str(Na[i]) + " " + str(N[Na[i]]) + " " + str(M[Ma[i]]))
    for i in range(M.shape[0]):
        diff += (Mv[i] - Nv[i]) ** 2
    diff = 6 * diff / (M.shape[0] ** 3 - M.shape[0])

    return 1 - diff


def w_jacc(M, N):
    mins = 0
    maxs = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            mins += min(M[i, j], N[i, j])
            maxs += max(M[i, j], N[i, j])
    return mins / maxs


def corrs(M, N):
    X = []
    for i in range(M.shape[0]):
        s = np.corrcoef(M[i], N[i])
        X.append(np.nan_to_num(s[0, 1]))
    return X


def relu(x):
    x[x < 0] = 0.
    return x


def make_mask(tsize, csize, label):
    mask = np.zeros([tsize, csize])
    if label:
        mask[:, 0:int(csize / 2)] = 1
    else:
        mask[:, int(csize / 2):] = 1
    return mask


data, labels, positions, masks = artificial_batching_patterned_space2(20000, 140, 50, seed=1988)
print(data[0, 0, 0:20])
data = np.moveaxis(data, 1, 2)
positions = positions.flatten().astype(np.int)
print('Original Data Shape:', data.shape)
print(masks.shape)


FinalData = data[18000:]
Labels = labels[18000:]

grads = np.load("./wandb/Sequence_Based_Models/MILC_100_models_saliencies/NPT_subj_0_trial_26.npy")

A = grads
# Stitch all the frames together
stiched_saliency = np.zeros((A.shape[0], components, samples_per_subject * sample_y))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        stiched_saliency[i, :, j * 20:j * 20 + sample_y] = grads[i, j, :, :]

grads = stiched_saliency

name = os.path.join(os.getcwd(), 'wandb', 'Sequence_Based_Models', 'MILC_100_models_saliencies')

print(grads.shape)

out_0 = []
out_1 = []
print(positions)
for i in range(grads.shape[0]):
    print(i)
    temp = FinalData[i]
    g = grads[i]
    x = np.abs(g)
    x = x / np.max(x)
    start = positions[i]

    temp_mask = np.zeros(temp.shape)
    temp_mask_discrete = np.zeros(temp.shape)

    label = Labels[i]
    features = temp_mask.shape[0]

    temp_mask[:, start:start + 20] = np.abs(temp[:, start:start + 20])  # make_mask(20, 20, labels[i])
    temp_mask_discrete[:, start:start+20] = 1

    if label:
        temp_mask[int(features / 2):, :] = 0
        temp_mask_discrete[int(features / 2):, :] = 0
    else:
        temp_mask[0:int(features / 2), :] = 0
        temp_mask_discrete[0:int(features / 2), :] = 0

    print(np.sum(temp_mask_discrete))
    robync = robyn_method(temp_mask_discrete, x)
    temp_mask = temp_mask / np.max(temp_mask)
    x_tst = x[:, start:start + 20]
    msk_tst = np.abs(temp[:, start:start + 20])
    T1, _ = scipy.stats.spearmanr(x.flatten(), temp_mask.flatten(), nan_policy="omit")
    T = scipy.stats.pearsonr(x.flatten(), temp_mask.flatten())

    # SR = spear_rank(msk_tst, x_tst)
    sr = T[0]
    spr = T1  # np.mean(T1)

    wj = w_jacc(x, temp_mask)
    # if np.isnan(np.array([sr,wj])).any() == :
    if labels[i] == 0:
        out_0.append(np.array([sr, wj, spr, robync]))
    else:
        out_1.append(np.array([sr, wj, spr, robync]))

np.save(name + "/out_0", out_0)
np.save(name + "/out_1", out_1)
print(np.array(out_0).shape)
print(np.mean(out_0, axis=0))

print(np.array(out_1).shape)
print(np.mean(out_1, axis=0))
