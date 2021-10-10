import torch
import numpy as np
from math import sqrt
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy
import h5py
import scipy.io as sio
from PIL import Image
import math
import h5py
import numpy as np
import cv2
from numpy import *
from PIL import Image
import math
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
import MPNCOV
from sklearn.metrics import mean_squared_error
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def mpsnr(x_true, x_pred):
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    x_true = x_true[:, :, :]
    for k in range(n_bands):
        x_true_k = x_true[:, :, k].reshape([-1])
        x_pred_k = x_pred[:, :, k, ].reshape([-1])
        
        MSE[k] = mean_squared_error(x_true_k, x_pred_k, )
        
        MAX_k = np.max(x_true_k)
        if MAX_k != 0:
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
        else:
            mask[k] = 0
    
    psnr = PSNR.sum() / mask.sum()
    mse = MSE.mean()
    return psnr


def ssim(x_true, x_pre):
    num = x_true.shape[2]
    ssimm = np.zeros(num)
    c1 = 0.0001
    c2 = 0.0009
    n = 0
    for x in range(x_true.shape[2]):
        z = np.reshape(x_pre[:, :, x], [-1])
        sa = np.reshape(x_true[:, :, x], [-1])
        y = [z, sa]
        cov = np.cov(y)
        oz = cov[0, 0]
        osa = cov[1, 1]
        ozsa = cov[0, 1]
        ez = np.mean(z)
        esa = np.mean(sa)
        ssimm[n] = ((2 * ez * esa + c1) * (2 * ozsa + c2)) / ((ez * ez + esa * esa + c1) * (oz + osa + c2))
        n = n + 1
    SSIM = np.mean(ssimm)
    return SSIM

def sam(x_true, x_pre):
    num = (x_true.shape[0]) * (x_true.shape[1])
    samm = np.zeros(num)
    n = 0
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            z = np.reshape(x_pre[x, y, :], [-1])
            sa = np.reshape(x_true[x, y, :], [-1])
            tem1 = np.dot(z, sa)
            tem2 = (np.linalg.norm(z)) * (np.linalg.norm(sa))
            A = (tem1 + 0.0001) / (tem2 + 0.0001)
            if A > 1:
                A = 1
            samm[n] = np.arccos(A)
            n = n + 1
    SAM = (np.mean(samm)) * 180 / np.pi
    return SAM


psnr = 0.0
psnrbic = 0.0
sssim = 0.0
ssimbic = 0.0
ssam = 0.0
sambic = 0.0


def se3Da(x):
    batchSize = x.data.shape[0]
    C = x.data.shape[1]
    depth = x.data.shape[2]
    h = x.data.shape[3]
    w = x.data.shape[4]
    M = h * w * depth
    x = x.reshape(batchSize, C, M)
    xT = x.transpose(1, 2)
    z = x.bmm(xT)
    aa = torch.sum(x, 2)
    aa = torch.unsqueeze(aa, 2)
    aa = aa.expand(-1, -1, M)
    c = aa.bmm(xT)
    cova = (1. / M) * z + (-1. / M / M) * c
    return cova


class se3D(nn.Module):
    def __init__(self):
        super(se3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv12 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = torch.nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        x1 = self.conv1(x1)
        x1 = self.conv12(x1)
        batch_size = x.data.shape[0]
        C = x.data.shape[1]
        x2 = se3Da(x)
        x2 = MPNCOV.SqrtmLayer(x2, 5)
        cov_mat_sum = torch.mean(x2, 1)
        x2 = cov_mat_sum.view(batch_size, C, 1, 1, 1)
        x2 = self.conv2(x2)
        x2 = self.conv22(x2)
        c = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(c)
        x3 = self.sigmoid(x3)
        x3 = x3 * x
        
        return x3


class Res2net1(nn.Module):
    def __init__(self):
        super(Res2net1, self).__init__()
        split = 4
        self.input = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, )
        self.output = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, )
        self.conv1 = nn.Conv3d(in_channels=64 // split, out_channels=64 // split, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=64 // split, out_channels=64 // split, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64 // split, out_channels=64 // split, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.senet = se3D()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out1, out2, out3, out4 = torch.chunk(out, 4, dim=1)
        out22 = self.relu(self.conv1(out2))
        # print(out22.shape)
        out33 = self.relu(self.conv2(out3 + out22))
        out44 = self.relu(self.conv3(out4 + out33))
        out = torch.cat((out1, out22, out33, out44), 1)
        out = self.output(out)
        out = residual + out
        out = self.senet(out)
        return out


class single_network1(nn.Module):
    def __init__(self):
        super(single_network1, self).__init__()
        
        self.a11 = Res2net1()
        self.a12 = Res2net1()
        self.a13 = Res2net1()
        self.a14 = Res2net1()
        self.a15 = Res2net1()
        self.a16 = Res2net1()
        self.a17 = Res2net1()
        self.a18 = Res2net1()
        self.a19 = Res2net1()
        self.a20 = Res2net1()
        self.a51 = Res2net1()
    
    def forward(self, x):
        residual = x
        out = self.a11(x)
        
        out = torch.add(out, residual)
        out = self.a12(out)
        
        out = torch.add(out, residual)
        out = self.a13(out)
        
        out = torch.add(out, residual)
        out = self.a14(out)
        
        out = torch.add(out, residual)
        out = self.a15(out)
        
        out = torch.add(out, residual)
        out = self.a16(out)
        
        out = torch.add(out, residual)
        out = self.a17(out)
        
        out = torch.add(out, residual)
        out = self.a18(out)
        
        out = torch.add(out, residual)
        out = self.a19(out)
        
        out = torch.add(out, residual)
        out = self.a20(out)
        
        out = torch.add(out, residual)
        out = self.a51(out)
        
        out = torch.add(out, residual)
        
        return out


class single_network2(nn.Module):
    def __init__(self):
        super(single_network2, self).__init__()
        
        self.a21 = Res2net1()
        self.a22 = Res2net1()
        self.a23 = Res2net1()
        self.a24 = Res2net1()
        self.a25 = Res2net1()
        self.a26 = Res2net1()
        self.a27 = Res2net1()
        self.a28 = Res2net1()
        self.a29 = Res2net1()
        self.a30 = Res2net1()
        
        self.a31 = Res2net1()
    
    def forward(self, x):
        residual = x
        out = self.a21(x)
        
        out = torch.add(out, residual)
        out = self.a22(out)
        
        out = torch.add(out, residual)
        out = self.a23(out)
        
        out = torch.add(out, residual)
        out = self.a24(out)
        
        out = torch.add(out, residual)
        out = self.a25(out)
        
        out = torch.add(out, residual)
        out = self.a26(out)
        
        out = torch.add(out, residual)
        out = self.a27(out)
        
        out = torch.add(out, residual)
        out = self.a28(out)
        
        out = torch.add(out, residual)
        out = self.a29(out)
        
        out = torch.add(out, residual)
        out = self.a30(out)
        
        out = torch.add(out, residual)
        out = self.a31(out)
        
        out = torch.add(out, residual)
        
        return out


class SRCNN(nn.Module):
    def __init__(self, scale):
        super(SRCNN, self).__init__()
        self.residual_layer1 = single_network1()
        self.residual_layer2 = single_network2()
        self.relu = nn.ReLU(inplace=True)
        self.input1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, )
        self.input2 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, )
        
        self.output1 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, )
        self.output2 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, )
        
        self.output11 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, )
        self.output21 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, )
        
        self.quanzhong1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, )
        self.quanzhong2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, )
        
        self.quanzhong3 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, )
        self.quanzhong4 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, )
        
        wn = lambda x: torch.nn.utils.weight_norm(x)
        
        tail1 = []
        tail1.append(wn(nn.ConvTranspose3d(64, 64, kernel_size=(3, 2 + scale, 2 + scale), stride=(1, scale, scale),
                                           padding=(1, 1, 1))))
        tail1.append(wn(nn.Conv3d(64, 16, 3, padding=3 // 2)))
        self.tail1 = nn.Sequential(*tail1)
        
        tail2 = []
        tail2.append(wn(nn.ConvTranspose3d(64, 64, kernel_size=(3, 2 + scale, 2 + scale), stride=(1, scale, scale),
                                           padding=(1, 1, 1))))
        tail2.append(wn(nn.Conv3d(64, 16, 3, padding=3 // 2)))
        self.tail2 = nn.Sequential(*tail2)
        self.senet = se3D()
    
    def forward(self, x):
        out = self.relu(self.input1(x))
        out = self.residual_layer1(out)
        out = self.tail1(out)
        out00 = self.output2(out)
        
        a1 = self.quanzhong1(out00)
        a2 = self.quanzhong2(a1)
        
        out1 = self.relu(self.input2(x))
        out1 = self.residual_layer2(out1)
        out1 = self.tail2(out1)
        out1 = self.output21(out1)
        
        b1 = self.quanzhong3(out1)
        b2 = self.quanzhong4(b1)
        
        c = torch.cat([a2, b2], dim=1)
        c = F.softmax(c, dim=1)
        c1, c2 = torch.chunk(c, 2, dim=1)
        out05 = torch.mul(c1, out00)
        out15 = torch.mul(c2, out1)
        out3 = torch.add(out05, out15)
        return out3


class subDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    
    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        
        return data, label


f = sio.loadmat('F:/LYT实验/数据集/100_4_36_144.mat')
input = f['dataa'].astype(np.float32)
label = f['label'].astype(np.float32)

input = np.reshape(input, [input.shape[0], input.shape[1], input.shape[2], input.shape[3], 1])
label = np.reshape(label, [label.shape[0], label.shape[1], label.shape[2], label.shape[3]])
input = numpy.transpose(input, (3, 4, 2, 1, 0))
label = numpy.transpose(label, (3, 0, 1, 2))

deal_dataset = subDataset(input, label)
test_loader = DataLoader.DataLoader(deal_dataset, batch_size=1, shuffle=False)

import time

for z in range(20, 50):
    print(z)
    
    GDRRN1 = SRCNN(4).cuda()
    
    GDRRN1 = torch.nn.DataParallel(GDRRN1)
    cudnn.benchmark = True
    GDRRN1.load_state_dict(torch.load('./2losses/2net_params%d.pkl' % (z)))
    branch_outputs = []
    psnr = 0.0
    psnrbic = 0.0
    sssim = 0.0
    ssimbic = 0.0
    ssam = 0.0
    sambic = 0.0
    
    start_time = time.time()
    output12 = []
    for i, (images, labels) in enumerate(test_loader):
        bic = images
        print(i)
        bic1 = np.transpose(bic, [0, 4, 3, 2, 1])
        bic2 = np.squeeze(bic1)
        images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)
        output11 = []
        label1 = labels.cpu()
        label2 = label1.data[0].numpy().astype(np.float32)
        with torch.no_grad():
            output = GDRRN1(images)
        output = output.cpu()
        result = output.data[0].numpy().astype(np.float32)
        result = np.transpose(result, [0, 3, 2, 1])
        NNB = np.squeeze(result)
        label3 = np.squeeze(label2)
        NNB = np.array(NNB)
        label3 = np.array(label3)
        bic2 = np.array(bic2)
        psnr = psnr + mpsnr(label3, NNB)
        sssim = sssim + ssim(label3, NNB)
        ssam = ssam + sam(label3, NNB)
        output12.append(result)
        result12 = np.concatenate(output12, axis=0)
    
    psnr = psnr / 7
    psnrbic = psnrbic / 7
    sssim = sssim / 7
    ssimbic = ssimbic / 7
    ssam = ssam / 7
    sambic = sambic / 7
    end_time = time.time() - start_time
    print(end_time / 7)
    print('psnr', psnr)
    #  print('psnrbic',psnrbic)
    print('sssim', sssim)
    #  print('ssimbic',ssimbic)
    print('ssam', ssam)
    psnr = 0
    psnrbic = 0
    sssim = 0
    ssimbic = 0
    ssam = 0
    sambic = 0
