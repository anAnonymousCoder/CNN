import numpy as np
import matplotlib.pyplot as plt
import torch


from TransformModule import fileUnpack

with open('Data/train_set.bin', 'rb') as f:
    train_dataset = fileUnpack(f)
    dBZ = [e[0] for e in train_dataset if e[2] > 0]

dBZ = np.array(dBZ)
reshapedBZ = []
for i in range(len(dBZ)):
    lineardBZ = np.array(dBZ[i][2:23, 2:23])
    lineardBZ = lineardBZ.reshape(-1)
    reshapedBZ.append(lineardBZ.tolist())

print(dBZ.shape)
