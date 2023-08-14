import os, torch
import numpy as np
#import pynvml
from torch import nn
from torch.nn import functional as F
from vis import vis_tool
import tifffile
from torch.utils.data import DataLoader
from Util import GetMemoryDataSetAndCrop
import math,time
from tqdm import tqdm

#from Net import Trainer
import Net
#import ParallelNet

'''
python -m visdom.server

在浏览器中打开：

http://localhost:8097/
'''


def CalcMeanStd(path):
    srcPath = path
    fileList = os.listdir(srcPath)
    fileNum = len(fileList)

    globalMean = 0
    globalStd = 0

    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        mean = np.mean(img)
        globalMean += mean
    globalMean /= fileNum


    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        img = img.astype(float)
        img -= globalMean
        sz = img.shape[0] * img.shape[1] * img.shape[2]
        globalStd += np.sum(img ** 2) / float(sz)
    globalStd = (globalStd / len(fileList)) ** (0.5)

    print(globalMean)
    print(globalStd)
    return globalMean,globalStd


def CalcMeanMax(path):
    srcPath = path
    fileList = os.listdir(srcPath)
    fileNum = len(fileList)

    maxVal = 0
    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        maxVal = np.maximum(maxVal, np.max(img))

    print(maxVal)

    globalMean = 0
    globalStd = 0

    for name in tqdm(fileList):
        img = tifffile.imread(os.path.join(srcPath, name))
        mean = np.mean(img)
        globalMean += mean
    globalMean /= fileNum
    print(globalMean)
    return globalMean,maxVal

env = 'AttentionUNetPerp'
globalDev = 'cuda:0'
globalDeviceID = 0
imgPath = "E:\Document\SuperRecon\ReconNet\sample\orig"
maskPath = "E:\Document\SuperRecon\ReconNet\sample/bin"

if __name__ == '__main__':
    lowMean,lowStd = 500,500#CalcMeanStd(imgPath)
    highMean, highStd = 4000,4000#CalcMeanStd(maskPath)
    train_dataset = GetMemoryDataSetAndCrop(imgPath, maskPath, [144, 144], 100, lowMean, lowStd, highMean, highStd)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=1)
    visdomable = True
    if visdomable == True:
        logger = vis_tool.Visualizer(env=env)
        logger.reinit(env=env)

    Net.logger = logger
    Net.lowMean = lowMean
    Net.lowStd = lowStd
    Net.highMean = highMean
    Net.highStd = highStd
    trainer = Net.Trainer(data_loader=train_loader, test_loader=None)

    time_start = time.time()
    trainer.Train(turn=500)
    time_end = time.time()
    print('totally time cost', time_end - time_start)
