import numpy as np
import os
import matplotlib.pyplot as plt
from source.utils import get_argparser
from scripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2


components = 50
time_points = 140
window_shift = 20
samples_per_subject = 7
sample_y = 20


X = np.zeros((2000, components, samples_per_subject * sample_y))
basename = "../wandb"
basename = os.path.join(basename, 'Sequence_Based_Models', 'MILC_100_models_saliencies')

for k in range(1, 101):
    print(k)
    # Read Saliency
    filename = os.path.join(basename, 'NPT_subj_0_trial_'+str(k)+'.npy')
    saliency = np.load(filename)
    print('Saliency {} loaded successfully...'.format(k))

    A = saliency
    print(saliency.shape)

    # Stitch all the frames together
    stiched_saliency = np.zeros((A.shape[0], components, samples_per_subject * sample_y))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            stiched_saliency[i, :, j * 20:j * 20 + sample_y] = saliency[i, j, :, :]

    saliency = stiched_saliency
    X += saliency

X = X/100
np.save(basename+"/new_dir_avg_sal_raw", X)



