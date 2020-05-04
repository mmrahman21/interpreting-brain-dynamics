
import numpy as np
import os
import matplotlib.pyplot as plt
from scripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2


components = 50
time_points = 140
window_shift = 20
samples_per_subject = 7
sample_y = 20

d = {}

basename = "../wandb"
basename = os.path.join(basename, 'Sequence_Based_Models', 'MILC_100_models_saliencies')
fnavgmap = os.path.join(basename, 'new_dir_avg_sal_raw.npy')
avg = np.load(fnavgmap)

filename = os.path.join(basename, 'NPT_subj_0_trial_'+str(1)+'.npy')
x = np.load(filename)
A = x
# Stitch all the frames together
stiched_saliency = np.zeros((A.shape[0], components, samples_per_subject * sample_y))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        stiched_saliency[i, :, j * 20:j * 20 + sample_y] = x[i, j, :, :]

x = stiched_saliency

dist_min = np.sum(np.sqrt(np.sum(np.sum(np.square(np.abs(avg - x)), axis=2), axis=1)))
target_file = filename

d['1'] = dist_min

for k in range(2, 101):
    print(k)
    filename = os.path.join(basename, 'NPT_subj_0_trial_'+str(k)+'.npy')
    x = np.load(filename)
    print('Saliency {} loaded successfully...'.format(k))

    A = x
    print(x.shape)

    # Stitch all the frames together
    stiched_saliency = np.zeros((A.shape[0], components, samples_per_subject * sample_y))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            stiched_saliency[i, :, j * 20:j * 20 + sample_y] = x[i, j, :, :]

    x = stiched_saliency

    sample_dist = np.sum(np.sqrt(np.sum(np.sum(np.square(np.abs(avg - x)), axis=2), axis=1)))

    d[k] = sample_dist

    if sample_dist < dist_min:
        dist_min = sample_dist
        target_file = filename
    print(d)

dict = sorted(d, key=d.get)
print(dict)
print(dist_min)
print(target_file)




