import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def get_data(path, batch_size, shuffle=True):
    videoIDs, videoSpeakers, videoLabels, videoText, \
        videoAudio, videoVisual, videoSentence, trainVid, \
        testVid = pickle.load(open(path, 'rb'), encoding='latin1')

    trainData = []
    trainTarget = []
    testData = []
    testTarget = []

    for vid in trainVid:
        trainData.append(videoAudio[vid])
        trainTarget.append(videoLabels[vid])

    for vid in testVid:
        testData.append(videoAudio[vid])
        testTarget.append(videoLabels[vid])

    trainData = np.array(trainData)
    trainTarget = np.array(trainTarget)
    testData = np.array(testData)
    testTarget = np.array(testTarget)

    trainData = torch.from_numpy(trainData).float()
    trainTarget = torch.from_numpy(trainTarget).long()
    testData = torch.from_numpy(testData).float()
    testTarget = torch.from_numpy(testTarget).long()

    trainDataset = MyDataset(trainData, trainTarget)
    testDataset = MyDataset(testData, testTarget)

    trainLoader = DataLoader(trainDataset, batch_size=batch_size,
                             shuffle=shuffle)
    testLoader = DataLoader(testDataset, batch_size=batch_size,
                            shuffle=shuffle)

    return trainLoader, testLoader


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_path, '../data/IEMOCAP_features.pkl')

    trainLoader, testLoader = get_data(path, 32)
