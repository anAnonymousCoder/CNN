import torch
import numpy as np
from TransformModule import fileUnpack

with open('Data/train_set.bin', 'rb') as f:
    train_dataset = fileUnpack(f)
    factor = [e[1] for e in train_dataset if e[2] > 0]  # factor为另外五个影响降雨量的因素


print(factor[0])
print(factor[1])