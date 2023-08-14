import os, tifffile
import Net, torch
import numpy as np
from models import AttentionUNet
import time
from tqdm import tqdm

img = tifffile.imread('origin_5X.tif')

minLowRange=[0,0,0]
minLowRange = minLowRange[-1::-1]
maxLowRange=[300,300,150]
maxLowRange = maxLowRange[-1::-1]#reverse
# readRange = [60, 60, 60]
zMinLowList = []
zMaxLowList = []
yMinLowList = []
yMaxLowList = []
xMinLowList = []
xMaxLowList = []

for k in range(minLowRange[0], maxLowRange[0] - 47, 32):
    zMinLowList.append(k)
    zMaxLowList.append(k+48)

zMinLowList.append(maxLowRange[0]-48)
zMaxLowList.append(maxLowRange[0])

for k in range(minLowRange[1], maxLowRange[1] - 47,32):
    #for k in range(minLowRange[1], maxLowRange[1] - 40,20):
    yMinLowList.append(k)
    yMaxLowList.append(k+48)
yMinLowList.append(maxLowRange[1]-48)
yMaxLowList.append(maxLowRange[1])

for k in range(minLowRange[2], maxLowRange[2] - 47,32):
    xMinLowList.append(k)
    xMaxLowList.append(k+48)
xMinLowList.append(maxLowRange[2]-48)
xMaxLowList.append(maxLowRange[2])

pretrained_net = AttentionUNet(in_channel = 1, num_class = 1)
pretrained_net.load_state_dict(torch.load('./saved_models/G_AB_31500.pth'))
pretrained_net = pretrained_net.cuda(0)
pretrained_net.eval()
torch.set_grad_enabled(False)
torch.cuda.empty_cache()

lowMeanVal = 500
lowStdVal = 500
highMeanVal = 4000
highStdVal = 4000
highImg = np.zeros((np.array(maxLowRange) - np.array(minLowRange))*3,dtype=np.uint16)
xBase = xMinLowList[0]
yBase = yMinLowList[0]
zBase = zMinLowList[0]
time_start = time.time()
for i in range(len(zMinLowList)):#TODO
    for j in range(len(yMinLowList)):
        for k in range(len(xMinLowList)):
            print('processing %d-%d, %d-%d %d-%d'%(xMinLowList[k], xMaxLowList[k],
                                                   yMinLowList[j], yMaxLowList[j],
                                                   zMinLowList[i], zMaxLowList[i]))
            lowImg = img[zMinLowList[i]: zMaxLowList[i],
                     yMinLowList[j]: yMaxLowList[j],
                     xMinLowList[k]: xMaxLowList[k]]

            lowImg = np.array(lowImg, dtype=np.float32)
            lowImg = (lowImg - lowMeanVal) / (lowStdVal)
            lowImg = np.expand_dims(lowImg, axis=0)
            lowImg = np.expand_dims(lowImg, axis=0)
            lowImg = torch.from_numpy(lowImg).float()
            lowImg = lowImg.cuda(0)
            pre2 = pretrained_net(lowImg)
            saveImg = pre2.cpu().data.numpy()[0, 0, :, :, :]
            saveImg *= lowStdVal
            saveImg += lowMeanVal
            saveImg = np.uint16(np.maximum(np.minimum(saveImg, 65535), 0))

            highImg[(zMinLowList[i]-zBase)*3+5:(zMinLowList[i]-zBase)*3+139,
            (yMinLowList[j]-yBase)*3+5:(yMinLowList[j]-yBase)*3+139,
            (xMinLowList[k]-xBase)*3+5:(xMinLowList[k]-xBase)*3+139] = saveImg[5:139, 5:139, 5:139]

time_end = time.time()
print('totally time cost', time_end - time_start)
tifffile.imwrite('att.tif', highImg)
