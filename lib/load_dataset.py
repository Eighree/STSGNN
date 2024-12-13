import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PeMSD7/pems07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join('../data/PeMSD3/pems03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'KnowAir':
        data_path = os.path.join('../data/KnowAir/KnowAir.npy')
        data = np.load(data_path)[:, :, 0]  # column 0: wind  column 3: Temperature  column -1: Pm2.5
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    if dataset == 'KnowAir':
        timeofday = np.zeros(shape=(data.shape[0], data.shape[1], 1))
        for i in range(data.shape[0]):
            timeofday[i, :, 0] = i + 1
        T = int(data.shape[0] / 8)

        dayofweek = np.zeros(shape=(data.shape[0], data.shape[1], 1))
        for i in range(T):
            dayofweek[i * 8:(i + 1) * 8, :, 0] = i + 1
        dayofweek[(i + 1) * 8:, :] = T + 1

    else:
        timeofday = np.zeros(shape=(data.shape[0],data.shape[1],1))
        for i in range(data.shape[0]):
            timeofday[i, :, 0] = i+1

        T = int(data.shape[0]/288)
        dayofweek = np.zeros(shape=(data.shape[0],data.shape[1],1))
        for i in range(T):
            dayofweek[i * 288:(i + 1) * 288, :, 0] = i+1
        dayofweek[(i + 1) * 288:, :] = T+1

    data = np.concatenate((data,dayofweek,timeofday),axis=-1)

    return data

