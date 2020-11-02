import math
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch

from TransformModule import fileUnpack

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

avg_dBZ = np.array(avg_dBZ)
RI = np.array(RI)
avg_tdBZ = np.array(avg_tdBZ)
tRI = np.array(tRI)

'''基线模型拟合train_set'''
plt.figure()
plt.title('Fitting result of train set')
plt.scatter(avg_dBZ, RI, s=2, c='b')
plt.xlabel('precipitation intensity(mm/h)')
plt.ylabel('dBZ($\mathregular{mm^3}$/$\mathregular{m^6}$)')
plt.ylim(ymax=90)
x = np.linspace(0, 40, 400)
y = 10 ** ((0.06459 * x) - 0.6459 * np.log10(30.08776))
plt.plot(x, y, c='r', label='fitting curve')
plt.legend()
plt.show()

'''基线模型拟合test_set'''
plt.figure()
plt.title('Fitting result of test set')
plt.scatter(avg_tdBZ, tRI, s=2, c='b')
plt.xlabel('precipitation intensity(mm/h)')
plt.ylabel('dBZ($\mathregular{mm^3}$/$\mathregular{m^6}$)')
plt.ylim(ymax=70)
x = np.linspace(0, 40, 400)
y = 10 ** ((0.0645921 * x) - 0.645921 * np.log10(80.0948))
plt.plot(x, y, c='r', label='fitting curve')
plt.legend()
plt.show()
