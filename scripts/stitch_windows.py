
import numpy as np
import torch

'''
components: number of features
samples_per_subject: number of windows
sample_y : number of time steps in each window
ws: amount of window shift

'''
def stitch_windows(saliency, components, samples_per_subject, sample_y, time_points, ws):

    stiched_saliency = np.zeros((saliency.shape[0], components, samples_per_subject * sample_y))
    for i in range(saliency.shape[0]):
        for j in range(saliency.shape[1]):
            stiched_saliency[i, :, j * 20:j * 20 + sample_y] = saliency[i, j, :, :]

    saliency = stiched_saliency

    avg_saliency = np.zeros((saliency.shape[0], components, time_points))

    if ws == 20:
        avg_saliency = saliency

    elif ws == 10:
        avg_saliency[:, :, 0:10] = saliency[:, :, 0:10]

        for j in range(samples_per_subject-1):
            a = saliency[:, :, 20*j+10:20*j+20]
            b = saliency[:, :, 20*(j+1):20*(j+1)+10]
            avg_saliency[:, :, 10*j+10:10*j+20] = (a + b)/2

        avg_saliency[:, :, time_points-10:time_points] = saliency[:, :, samples_per_subject*sample_y-10:samples_per_subject*sample_y]

    else:
        for i in range(saliency.shape[0]):
            for j in range(time_points):
                L = []
                if j < 20:
                    index = j
                else:
                    index = 19 + (j - 19) * 20

                L.append(index)

                s = saliency[i, :, index]
                count = 1
                block = 1
                iteration = min(19, j)
                for k in range(0, iteration, 1):
                    if index + block * 20 - (k + 1) < samples_per_subject * sample_y:
                        s = s + saliency[i, :, index + block * 20 - (k + 1)]
                        L.append(index + block * 20 - (k + 1))
                        count = count + 1
                        block = block + 1
                    else:
                        break
                avg_saliency[i, :, j] = s/count
                # print('Count =', count, ' and Indices =', L)

    return avg_saliency

