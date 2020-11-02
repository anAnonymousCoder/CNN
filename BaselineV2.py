import numpy as np
import matplotlib.pyplot as plt
import math
import torch

from fractions import Fraction
from scipy.optimize import leastsq

from TransformModule import fileUnpack

'''使用最小二乘法确定Z-R关系V 2.0'''
'''本方法中使用中心点dBZ值'''
'''将Z-R模型估算的降水强度与实际降水强度进行比较'''

with open('Data/train_set.bin', 'rb') as f:
    train_dataset = fileUnpack(f)
    dBZ = [e[0] for e in train_dataset if e[2] > 0]
    avg_dBZ = [float(torch.Tensor(e).mean()) for e in dBZ]  # avg_dBZ中存储着637个25*25大小的dBZ矩阵的平均值
    RI = [e[2] * 10 for e in train_dataset if e[2] > 0]  # RI中保存着降雨强度

with open('Data/test_set.bin', 'rb') as f:
    test_dataset = fileUnpack(f)
    tdBZ = [e[0] for e in test_dataset if e[2] > 0]
    avg_tdBZ = [float(torch.Tensor(e).mean()) for e in tdBZ]
    tRI = [e[2] * 10 for e in test_dataset if e[2] > 0]


dBZ = np.array(dBZ)
print(dBZ[0][12][12])
center_dBZ = [dBZ[i][12][12] for i in range(len(dBZ))]  # 获取中心区域dBZ
'''center_dBZ = [dBZ[i] for i in range(312, len(dBZ), 625)]
print(len(center_dBZ))'''

tdBZ = np.array(tdBZ)
center_tdBZ = [tdBZ[i][12][12] for i in range(len(tdBZ))]  # 获取中心区域tdBZ


def func(p, x):  # 拟合函数
    a, b = p
    return 10 * (b * np.log10(x) + np.log10(a))  # 通过R求dBZ的公式


def error(p, x, y):  # 损失函数
    ret = func(p, x) - y
    ret = np.append(ret, np.sqrt(0.0001) * p)
    return ret


p0 = [1e-5, 1.0]  # 参数a,b初始值

RI = np.array(RI)
center_dBZ = np.array(center_dBZ)

Para = leastsq(error, p0, args=(RI, center_dBZ), maxfev=10000)  # 迭代一万次，寻找dBZ和RI的关系

a, b = Para[0]
print("a=", a, "b=", b)


a = round(a, 6)
b = round(b, 6)  # a,b各保留六位小数

'''print(avg_tdBZ.index(max(avg_tdBZ)))
print(avg_tdBZ[44])
print(tRI[44])

print(tRI.index(max(tRI)))
print(avg_tdBZ[259])
print(tRI[259])
print(math.pow(10, ((0.1131 * 30.74) - 1.131 * np.log10(67.3))))'''

tRI = np.array(tRI)
center_tdBZ = np.array(center_tdBZ)

est_RI = []  # 根据baseline model计算降雨强度估测值
for ct in center_tdBZ:
    est_RI.append(np.power(10, ((0.06459 * ct) - 0.6459 * np.log10(a))))
est_RI = np.array(est_RI)


plt.figure()  # 绘制观测值与估测值的图像
plt.scatter(tRI, est_RI, c='', marker='s', edgecolors='b', s=10)
plt.xlabel("observation(mm/h)")
plt.ylabel("estimation(mm/h)")
x = np.linspace(0, 60, 1000)
y = x
plt.plot(x, y, c='k', linestyle=':', label='y = x')
plt.title('The estimation of Baseline Model')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.legend()
plt.show()


Number = np.arange(1, len(tRI) + 1, 1)  # 观测值与估计值按样本排列的图像
plt.figure(figsize=(10, 4))
plt.plot(Number, tRI, c='b', label='observation', linewidth=1, alpha=0.75)
plt.plot(Number, est_RI, c='r', linestyle='--', label='estimation', linewidth=1)
plt.xlabel('Sample Number')
plt.ylabel('Precipitation intensity(mm/h)')
plt.legend()
plt.show()


def get_MSE(X, Y):  # 计算均方误差
    return sum([(x - y) ** 2 for x, y in zip(X, Y)]) / len(X)


print("MSE:", get_MSE(tRI, est_RI))  # 打印均方误差
