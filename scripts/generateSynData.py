'''
This code is used to generate Noah's Synthetic data
'''

import numpy as np
import matplotlib.pyplot as plt


def artificial_batching_patterned_space2(samples, t_steps, features, p_steps=10, seed=None):
    if seed != None:
        np.random.seed(seed)
    X = np.zeros([samples, t_steps, features])
    L = np.zeros([samples])
    start_positions = np.zeros([samples])
    masks = np.zeros([samples,p_steps,features])
    for i in range(samples):
        mask = np.zeros([p_steps, features])
        #0,17 ; 27,47
        start = np.random.randint(0,t_steps-p_steps)
        start_positions[i] = start
        x = np.random.normal(0, 1, [1, t_steps, features])
        label = np.random.randint(0, 2)
        lift = np.random.normal(1, 1,[p_steps,features])#np.random.normal(0, 1, [p_steps, features])
        X[i,:,:] = x
        if label:
            mask[:,0:int(features/2)] = 1
        else:
            mask[:,int(features/2):] = 1
        lift = lift*mask
        X[i,start:start+p_steps, :] += lift
        masks[i,:,:] = lift
        L[i] = int(label)
    return X, L, start_positions, masks

def artificial_batching_patterned_space(samples, t_steps, features, p_steps=10, seed=None):
    if seed != None:
        np.random.seed(seed)
    X = np.zeros([samples, t_steps, features])
    L = np.zeros([samples])
    start_positions = np.zeros([samples])
    masks = np.zeros([samples,p_steps,features])
    for i in range(samples):
        mask = np.zeros([p_steps, features])
        #0,17 ; 27,47
        start = np.random.randint(0,t_steps-p_steps)
        start_positions[i] = start
        x = np.random.normal(0, 1, [1, t_steps, features])
        label = np.random.randint(0, 2)
        lift = np.random.normal(1, 1,[p_steps,features]) #np.random.normal(0, 1, [p_steps, features])
        X[i,:,:] = x
        if label:
            mask[:,0:int(features/2)] = 1
        else:
            mask[:,int(features/2):] = 1
        lift = lift*mask
        X[i,start:start+p_steps, :] = lift
        masks[i,:,:] = lift
        L[i] = int(label)
    return X, L, start_positions, masks

# Data , L, start_positions, masks = artificial_batching_patterned_space2(60000, 140, 50)
# print(Data[0, 0, 0:20])
# print(Data.shape)
# print(L.shape)
# print(start_positions.shape)
# print(masks.shape)
# fig = plt.figure()
# Data = np.moveaxis(Data, 1, 2)
# print(Data.shape)

# np.save('../Noah Synthetic Data/Data.npy', Data)
# np.save('../Noah Synthetic Data/Labels.npy', L)
# np.save('../Noah Synthetic Data/start_positions.npy', start_positions)
# np.save('../Noah Synthetic Data/masks.npy', masks)
# print('Data Saved Successfully')
# color_map = plt.cm.get_cmap('gray')
# reversed_color_map = color_map.reversed()
# plt.imshow(Data[0], interpolation='nearest', aspect='auto', cmap='Reds')
# plt.show()