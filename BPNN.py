import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import numpy as np

import matplotlib.pyplot as plt

from TransformModule import fileUnpack

LOAD_MODEL = False


# STOP_TRAIN_ROUND = 3
# 如果连续3个epoch testloss都没下降 就终止训练


class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        # in 1,21,21
        self.layer1 = nn.Sequential(
            nn.Linear(446, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            # 随机纠正线性单元
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, dBZ, factor):
        dBZ = torch.cat((dBZ, factor), dim=1)
        dBZ = self.layer1(dBZ)
        dBZ = self.layer2(dBZ)
        dBZ = self.layer3(dBZ)
        out = self.layer4(dBZ)
        return out


learning_rate = 1e-5  # 设置学习率为0.00001
num_epochs = 10000
cuda_available = torch.cuda.is_available()  # 使用cuda加速

# 加载训练集数据
with open('Data/train_set.bin', 'rb') as f:
    train_dataset = fileUnpack(f)
    dBZ = [e[0] for e in train_dataset if e[2] > 0]  # dBZ为多个21*21的dBZ矩阵
    factor = [e[1] for e in train_dataset if e[2] > 0]  # factor为另外五个影响降雨量的因素
    RI = torch.Tensor([e[2] * 10 for e in train_dataset if e[2] > 0])  # RI(Rainfall Intensity)为降雨强度的真实值
    avg_dBZ = [float(torch.Tensor(e).mean()) for e in dBZ]  # avg_dBZ为dBZ矩阵的均值
    real_RI = RI  # real_RI为降雨强度的真实值
    dBZ = np.array(dBZ)
    reshapedBZ = []
    for i in range(len(dBZ)):
        lineardBZ = np.array(dBZ[i][2:23, 2:23])
        lineardBZ = lineardBZ.reshape(-1)
        reshapedBZ.append(lineardBZ.tolist())  # 将25*25重塑为1*(21*21)
    dBZ = torch.Tensor(reshapedBZ)  # 将dBZ处理为张量
    factor = torch.Tensor(factor)  # dBZ = dBZ.view(dBZ.size(0), -1)
    RI = torch.reshape(RI, [RI.shape[0], 1])  # 将RI转换为多行单列的张量

with open('Data/test_set.bin', 'rb') as f:
    test_dataset = fileUnpack(f)
    tdBZ = [e[0] for e in test_dataset if e[2] > 0]
    tfactor = [e[1] for e in test_dataset if e[2] > 0]
    tRI = torch.Tensor([e[2] * 10 for e in test_dataset if e[2] > 0])
    avg_tdBZ = [float(torch.Tensor(e).mean()) for e in tdBZ]
    real_tRI = tRI
    tdBZ = np.array(tdBZ)
    treshapedBZ = []
    for i in range(len(tdBZ)):
        lineardBZ = np.array(tdBZ[i][2:23, 2:23])
        lineardBZ = lineardBZ.reshape(-1)
        treshapedBZ.append(lineardBZ.tolist())  # 将25*25重塑为21*21
    tdBZ = torch.Tensor(treshapedBZ)
    tfactor = torch.Tensor(tfactor)
    # tdBZ = tdBZ.view(tdBZ.size(0), -1)
    tRI = torch.reshape(tRI, [tRI.shape[0], 1])

if LOAD_MODEL:
    model = torch.load('model/modelbp.pth')
else:
    model = BPNet()

if cuda_available:
    model = model.cuda()  # 将训练放在GPU上进行

criterion = nn.MSELoss()  # 定义损失函数为MSE（最小平方误差函数）
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001)  # 定义迭代优化算法为随机梯度下降算法

plt.ion()  # 使图像变为交互模式
plt.figure()  # 创建画布

loss_train = []  # 存储训练损失的数组
loss_test = []

if not LOAD_MODEL:  # 训练
    plt.scatter(avg_dBZ, real_RI, c='b', s=3)  # 绘制散点图

    eval_loss = 0

    if cuda_available:  # 将训练转移到GPU上进行
        inputs = Variable(dBZ).cuda()
        factors = Variable(factor).cuda()
        values = Variable(RI).cuda()
    else:
        inputs = Variable(dBZ)
        factors = Variable(factor)
        values = Variable(RI)
    # inputs = inputs.view(inputs.size(0), 1, inputs.size(1), inputs.size(2))

    if cuda_available:
        tinputs = Variable(tdBZ).cuda()
        tfactors = Variable(tfactor).cuda()
        tvalues = Variable(tRI).cuda()
    else:
        tinputs = Variable(tdBZ)
        tfactors = Variable(tfactor)
        tvalues = Variable(tRI)
    # tinputs = tinputs.view(tinputs.size(0), 1, tinputs.size(1), tinputs.size(2))

    for epoch in range(num_epochs):
        # forward前向传播计算网络结构输出结果
        out = model(inputs, factors)
        loss = criterion(out, values)  # 计算损失函数
        # backward后向传播更新参数
        optimizer.zero_grad()  # 将上次迭代计算的梯度值清0，防止梯度爆炸
        loss.backward()  # 反向传播，计算梯度值
        optimizer.step()  # 更新权值参数

        if not (epoch + 1) % 100 and False:
            plt.cla()  # 清除当前画布中活动的axis
            pred = model(inputs, factors).cpu().data.numpy()  # 计算降水量预测值
            plt.scatter(avg_dBZ, real_RI, c='b', s=3, label='real data')
            # 绘制散点图
            plt.scatter(avg_dBZ, pred, c='r', s=3, label='predicted values')
            plt.legend()  # 给图像加上图例，以便更好地看懂图像
            plt.pause(0.01)  # 继续绘图并且鼠标可以进行交互
            print("Epoch[{}/{}], loss: {:.6f}"
                  .format(epoch + 1, num_epochs, loss.item()
                          ))

        if not (epoch + 1) % 100:
            model.eval()  # 测试数据

            loss_train.append(loss.item())
            out = model(tinputs, tfactors)
            loss = criterion(out, tvalues)  # eval模式不会更新权值参数
            loss_test.append(loss.item())

            plt.cla()
            xz = np.arange(0, len(loss_train) * 100, 100)
            # 绘图，横轴起点为0，终点为len(loss_train)*100，步长为100
            plt.plot(xz, loss_train, c='b', label="train dataset")
            plt.plot(xz, loss_test, c='r', label="test dataset")
            plt.legend(loc=1)  # 在图像右上角显示图例
            plt.pause(0.01)

            print('Epoch[%d/%d], Train Loss: %.6f, Test Loss: %.6f' % (
                epoch + 1, num_epochs, loss_train[-1], loss_test[-1]
            ))
            model.train()

    torch.save(model, 'model/modelbp.pth')  # 保存训练后的模型

input()  # 控制台输入任意字符，显示估算值与观测值比较图像
'''plt.cla()
plt.ioff()

xz = np.arange(0,len(loss_train))
plt.plot(xz, loss_train, c='b')
plt.plot(xz, loss_test, c='r')
plt.show()'''

if cuda_available:
    tinputs = Variable(tdBZ).cuda()
    tvalues = Variable(tRI).cuda()
    tfactors = Variable(tfactor).cuda()
else:
    tinputs = Variable(tdBZ)
    tvalues = Variable(tRI)
    tfactors = Variable(tfactor)
# tinputs = tinputs.view(tinputs.size(0), 1, tinputs.size(1), tinputs.size(2))
plt.ioff()
plt.cla()
model.eval()  # 测试数据
out = model(tinputs, tfactors)
loss = criterion(out, tvalues)
out = out.cpu().data
out = out.view(out.size(0)).numpy().tolist()
real_tRI = real_tRI.numpy().tolist()
avg_tdBZ = avg_tdBZ[::1]
real_tRI = real_tRI[::1]
out = out[::1]
'''for i in range(len(out)):
    plt.scatter([avg_tdBZ[i]],[real_tRI[i]], c='b', s=3)
    plt.scatter([avg_tdBZ[i]],[out[i]], c='r', s=3)
    plt.plot([avg_tdBZ[i], avg_tdBZ[i]], [real_tRI[i], out[i]], c='g', linewidth=1)'''

plt.scatter(avg_tdBZ, real_tRI, c='b', s=3, label='real data')
plt.scatter(avg_tdBZ, out, c='r', s=3, label='predicted values')
'''plt.scatter(out, real_tRI, c='b', s=3)
plt.xlabel("estimation")
plt.ylabel("observation")
x = np.linspace(0, 50, 100)
y = x
plt.plot(x, y, c='r')'''
plt.legend()
plt.show()
print('Test Loss: %.6f' % (
    loss.item()
))

print(len(loss_train))
plt.figure()
xz = np.arange(0, len(loss_train) * 100, 100)
# 绘图，横轴起点为0，终点为len(loss_train)*100，步长为100
plt.plot(xz, loss_train, c='b', label="train dataset")
plt.plot(xz, loss_test, c='r', label="test dataset")
plt.legend()
plt.show()

import pickle as pkl

with open('Result/BPResult/dbzbp0.pkl', 'wb') as dbzbp0:
    pkl.dump(avg_tdBZ, dbzbp0)  # 平均dBZ值保存在pkl文件中
with open('Result/BPResult/obbp0.pkl', 'wb') as obbp0:
    pkl.dump(real_tRI, obbp0)  # 保存真实降水强度
with open('Result/BPResult/estbp0.pkl', 'wb') as estbp0:
    pkl.dump(out, estbp0)  # 保存估算降水强度
with open('Result/BPResult/trainbp0.pkl', 'wb') as trainbp0:
    pkl.dump(loss_train, trainbp0)  # 保存真实降水强度
with open('Result/BPResult/testbp0.pkl', 'wb') as testbp0:
    pkl.dump(loss_test, testbp0)  # 保存估算降水强度
