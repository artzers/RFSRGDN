import os, torch
import numpy as np
from torch import nn
import tifffile
from tqdm import tqdm



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def prepare(dev, *args):
    # print(dev)
    device = torch.device(dev)
    if dev == 'cpu':
        device = torch.device('cpu')
    return [a.to(device) for a in args]


def calc_psnr(sr, hr, scale):
    diff = (sr - hr)
    # shave = scale + 6
    # valid = diff[..., shave:-shave, shave:-shave,:]#2，2，1
    # mse = valid.pow(2).mean()
    mse = np.mean(diff * diff) + 0.0001
    return -10 * np.log10(mse / (4095 ** 2))


def RestoreNetImg(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    maxVal = np.max(rImg)
    minVal = np.min(rImg)
    rImg = 255./(maxVal - minVal+1) * (rImg - minVal)
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

def RestoreNetImgV2(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

class WDSRBBlock3D(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSRBBlock3D, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv3d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv3d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv3d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # res = self.body(x) * self.res_scale
        # res += x
        res = self.body(x) + x
        return res

class PixelUpsampler3D(nn.Module):
    def __init__(self,
                 upscaleFactor,
                 # conv=default_conv3d,
                 # n_feats=32,
                 # kernel_size=3,
                 # bias=True
                 ):
        super(PixelUpsampler3D, self).__init__()
        self.scaleFactor = upscaleFactor

    def PixelShuffle(self, input, upscaleFactor):
        batchSize, channels, inDepth, inHeight, inWidth = input.size()
        channels //= upscaleFactor[0] * upscaleFactor[1] * upscaleFactor[2]
        outDepth = inDepth * upscaleFactor[0]
        outHeight = inHeight * upscaleFactor[1]
        outWidth = inWidth * upscaleFactor[2]
        inputView = input.contiguous().view(
            batchSize, channels, upscaleFactor[0], upscaleFactor[1], upscaleFactor[2], inDepth,
            inHeight, inWidth)
        shuffleOut = inputView.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffleOut.view(batchSize, channels, outDepth, outHeight, outWidth)

    def forward(self, x):
        # x = self.conv(x)
        up = self.PixelShuffle(x, self.scaleFactor)
        return up

class GetMemoryDataSetAndCrop:
    def __init__(self, imgDir, maskDir, cropSize, epoch, lowMean, lowStd, highMean, highStd):
        self.imgDir = imgDir
        self.maskDir = maskDir

        self.epoch = epoch

        self.lowMean = lowMean
        self.lowStd = lowStd
        self.highMean = highMean
        self.highStd = highStd

        self.imgFileList = []
        self.maskFileList = []

        self.imgList = []
        self.maskList = []

        self.beg = [0, 0, 0]
        self.cropSz = cropSize

        for file in os.listdir(self.imgDir):
            if file.endswith('.tif'):
                self.imgFileList.append(file)
        self.imgFileList.sort()

        for file in os.listdir(self.maskDir):
            if file.endswith('.tif'):
                self.maskFileList.append(file)
        self.maskFileList.sort()

        # if len(self.imgFileList) != len(self.maskFileList):
        #     self.check = False

        for k in tqdm(range(len(self.imgFileList))):
            imgName = os.path.join(self.imgDir, self.imgFileList[k])
            img3d = tifffile.imread(imgName)
            self.imgList.append(img3d)
            # zlen = img3d.shape[0]
            # for jj in range(zlen):
            #     img = np.expand_dims(img3d[jj,:,:], axis=0)
            #     self.imgList.append(img)

        for k in tqdm(range(len(self.maskFileList))): #tqdm(range(1)): #
            maskName = os.path.join(self.maskDir, self.maskFileList[k])
            mask3d = tifffile.imread(maskName)
            self.maskList.append(mask3d)

        self.gid = 1

        self.zLen2 = 144
        self.zLen = self.zLen2//3

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.imgFileList)

    def __len__(self):
        return self.epoch

    def len(self):
        return self.epoch

    def __getitem__(self, ind):

        imgList = np.zeros((self.zLen, self.cropSz[0], self.cropSz[1]),
                           self.maskList[0].dtype)
        resMask = np.zeros((self.zLen * 3, self.cropSz[0], self.cropSz[1]),
                           self.maskList[0].dtype)

        # sz = self.maskList[0].shape

        ind = np.random.randint(0, len(self.maskList))
        sz = self.maskList[ind].shape
        self.beg[0] = np.random.randint(0, sz[0] - self.zLen2 - 1)  # z
        self.beg[1] = np.random.randint(0, sz[1] - self.cropSz[0] - 1)  # z
        self.beg[2] = np.random.randint(0, sz[2] - self.cropSz[1] - 1)

        # for k in range(self.zLen * 3):
        resMask = (self.maskList[ind][self.beg[0]:self.beg[0] + self.zLen2,
                   self.beg[1]:self.beg[1] + self.cropSz[0],
                   self.beg[2]:self.beg[2] + self.cropSz[1]]).astype(float)

        resMask = (resMask - self.highMean) / self.highStd

        ind = np.random.randint(0, len(self.imgList))
        sz = self.imgList[ind].shape
        while True:
            self.beg[0] = np.random.randint(0, sz[0] - self.zLen - 2)  # z
            self.beg[1] = np.random.randint(0, sz[1] - self.cropSz[0]//3 - 2)  # z
            self.beg[2] = np.random.randint(0, sz[2] - self.cropSz[1]//3 - 2)

            imgList = (self.imgList[ind][self.beg[0]:self.beg[0] + self.zLen,
                       self.beg[1]:self.beg[1] + self.cropSz[0]//3,
                       self.beg[2]:self.beg[2] + self.cropSz[1]//3]).astype(float)

            imgList = (imgList - self.lowMean) / self.lowStd

            if np.std(imgList) > 0.6:
                break

        rid = np.random.randint(0, 3)
        if rid == 0:
            pass  # return lrImg, midImg, hrImg
        if rid == 1:
            # for k in range(15):
            #     imgList[k] = imgList[k][:,::-1,:]
            imgList = imgList[:, ::-1, :]
            resMask = resMask[:, ::-1, :]
        if rid == 2:
            # for k in range(15):
            #     imgList[k] = imgList[k][:, :, ::-1]
            imgList = imgList[:, :, ::-1]
            resMask = resMask[:, :, ::-1]

        imgList = np.expand_dims(imgList, axis=0)
        imgList = torch.from_numpy(imgList.copy()).float()
        resMask = np.expand_dims(resMask, axis=0)
        resMask = torch.from_numpy(resMask.copy()).float()

        return imgList, resMask
