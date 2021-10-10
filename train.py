import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
import numpy
import h5py
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import MPNCOV

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        return out3, out00, out1


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


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = numpy.array(hf.get('data'))
        label = numpy.array(hf.get('label'))
        train_data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1])
        train_label = np.reshape(label, [label.shape[0], label.shape[1], label.shape[2], label.shape[3], 1])
        train_data = numpy.transpose(train_data, (0, 4, 1, 2, 3))
        train_label = numpy.transpose(train_label, (0, 4, 1, 2, 3))
        return train_data, train_label


def tiduloss(a, b):
    a3 = gradients(a)
    b3 = gradients(b)
    c = torch.abs(a3 - b3)
    c = torch.mean(c)
    return c


def gradients(image):
    dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
    return dx


def train_fun2(EPOCH):
    gpus = [0]
    cuda_gpu = torch.cuda.is_available()
    my_cnn = SRCNN(4)
    if (cuda_gpu):
        my_cnn = torch.nn.DataParallel(my_cnn, device_ids=gpus).cuda()
    optimizer = optim.Adam(my_cnn.parameters(), lr=0.0005)
    
    train_data, train_label = read_training_data("F:/LYT实验/数据集/旋转100tongdao4bei9_36.h5")
    deal_dataset = subDataset(train_data, train_label)
    train_loader = DataLoader.DataLoader(deal_dataset, batch_size=16, shuffle=True)  # 16
    loss_list = []
    loss = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)
    for eopch in range(EPOCH):
        for step, data in enumerate(train_loader):
            train_data, train_label = data
            train_data, train_label = Variable(train_data.cuda()), Variable(train_label.cuda())
            output_f, output_1, output_2 = my_cnn(train_data)
            loss_f = loss(output_f, train_label)
            loss_1 = loss(output_1, train_label)
            loss_2 = loss(output_2, train_label)
            loss_4 = loss(output_2, output_1)
            loss_TD = tiduloss(output_f, train_label)
            loss_3 = loss_f + 0.5 * loss_2 + 0.5 * loss_1 + 0.5 * loss_4 + loss_TD
            if step % 10 == 0:
                print('Epoch:', eopch, 'Step: ', step,
                      'loss_f: {:.6f}\t'.format(float(loss_f)),
                      'loss_2: {:.6f}\t'.format(float(loss_2)),
                      'loss_1: {:.6f}\t'.format(float(loss_1)),
                      'loss_4: {:.6f}\t'.format(float(loss_4)),
                      'loss_td: {:.6f}\t'.format(float(loss_TD)),
                      'loss_3: {:.6f}\t'.format(float(loss_3)))
                print("lr", optimizer.state_dict()['param_groups'][0]['lr'])
                loss_list.append(float(loss_f))
                torch.save(my_cnn.state_dict(), './0.5losses/2net_params%d.pkl' % (eopch))
            optimizer.zero_grad()
            loss_3.backward()
            optimizer.step()
        scheduler.step()
    return loss_list


def train_fun3(EPOCH):
    gpus = [0]
    cuda_gpu = torch.cuda.is_available()
    my_cnn = SRCNN(3)
    if (cuda_gpu):
        my_cnn = torch.nn.DataParallel(my_cnn, device_ids=gpus).cuda()
    optimizer = optim.Adam(my_cnn.parameters(), lr=0.0005)
    
    train_data, train_label = read_training_data("旋转100tongdao4bei12_36.h5")
    deal_dataset = subDataset(train_data, train_label)
    train_loader = DataLoader.DataLoader(deal_dataset, batch_size=12, shuffle=True)  # 16
    loss_list = []
    loss = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)
    for eopch in range(EPOCH):
        for step, data in enumerate(train_loader):
            train_data, train_label = data
            train_data, train_label = Variable(train_data.cuda()), Variable(train_label.cuda())
            output_f, output_1, output_2 = my_cnn(train_data)
            loss_f = loss(output_f, train_label)
            loss_1 = loss(output_1, train_label)
            loss_2 = loss(output_2, train_label)
            loss_4 = loss(output_2, output_1)
            loss_TD = tiduloss(output_f, train_label)
            loss_3 = loss_f + loss_2 + loss_1 + loss_4 + loss_TD
            
            if step % 10 == 0:
                print('Epoch:', eopch, 'Step: ', step,
                      'loss_f: {:.6f}\t'.format(float(loss_f)),
                      'loss_2: {:.6f}\t'.format(float(loss_2)),
                      'loss_1: {:.6f}\t'.format(float(loss_1)),
                      'loss_4: {:.6f}\t'.format(float(loss_4)),
                      'loss_td: {:.6f}\t'.format(float(loss_TD)),
                      'loss_3: {:.6f}\t'.format(float(loss_3)))
                print("lr", optimizer.state_dict()['param_groups'][0]['lr'])
                loss_list.append(float(loss_f))
                torch.save(my_cnn.state_dict(), './新3倍/3net_params%d.pkl' % (eopch))
            optimizer.zero_grad()
            loss_3.backward()
            optimizer.step()
        scheduler.step()
    return loss_list


def train_fun4(EPOCH):
    gpus = [0]
    cuda_gpu = torch.cuda.is_available()
    my_cnn = SRCNN(4)
    if (cuda_gpu):
        my_cnn = torch.nn.DataParallel(my_cnn, device_ids=gpus).cuda()
    optimizer = optim.Adam(my_cnn.parameters(), lr=0.0005)
    
    train_data, train_label = read_training_data("旋转100tongdao4bei9_36.h5")
    print(train_data.shape, train_label.shape)
    deal_dataset = subDataset(train_data, train_label)
    train_loader = DataLoader.DataLoader(deal_dataset, batch_size=32, shuffle=True)  # 16
    loss_list = []
    loss = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)
    
    for eopch in range(EPOCH):
        for step, data in enumerate(train_loader):
            train_data, train_label = data
            train_data, train_label = Variable(train_data.cuda()), Variable(train_label.cuda())
            output_f, output_1, output_2 = my_cnn(train_data)
            loss_f = loss(output_f, train_label)
            loss_1 = loss(output_1, train_label)
            loss_2 = loss(output_2, train_label)
            loss_4 = loss(output_2, output_1)
            loss_TD = tiduloss(output_f, train_label)
            loss_3 = loss_f + loss_2 + loss_1 + loss_4 + loss_TD
            
            if step % 10 == 0:
                print('Epoch:', eopch, 'Step: ', step,
                      'loss_f: {:.6f}\t'.format(float(loss_f)),
                      'loss_2: {:.6f}\t'.format(float(loss_2)),
                      'loss_1: {:.6f}\t'.format(float(loss_1)),
                      'loss_4: {:.6f}\t'.format(float(loss_4)),
                      'loss_td: {:.6f}\t'.format(float(loss_TD)),
                      'loss_3: {:.6f}\t'.format(float(loss_3)))
                loss_list.append(float(loss_f))
                torch.save(my_cnn.state_dict(), './新4倍1/4net_params%d.pkl' % (eopch))
            optimizer.zero_grad()
            loss_3.backward()
            optimizer.step()
        scheduler.step()
    return loss_list


if __name__ == "__main__":
    import gc
    
    gc.collect()
    train_fun2(100)
    gc.collect()
    train_fun3(100)
    gc.collect()
    train_fun2(50)
