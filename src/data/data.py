import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


def mnist():
    # training data is separated into multiple files - load and concatenate.
    traindata = np.load('C:/MLOPS/dtu_mlops/data/corruptmnist/train_0.npz')
    images = traindata['images']
    labels = traindata['labels']

    for i in range(1,4):
        trainset = np.load(f'C:/MLOPS/dtu_mlops/data/corruptmnist/train_{i}.npz')
        images = np.concatenate((images, trainset['images']))
        labels = np.concatenate((labels, trainset['labels']))
    
    trainset = TensorDataset(torch.Tensor(images), torch.LongTensor(labels))
    
    # load and transform test
    testdata = np.load('C:/MLOPS/dtu_mlops/data/corruptmnist/test.npz')
    images = testdata['images']
    labels = testdata['labels']
    testset = TensorDataset(torch.Tensor(images), torch.LongTensor(labels))
    
    return trainset, testset
