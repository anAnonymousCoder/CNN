import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

'''
分析CNN估测的降水强度和实际降水强度的偏差
'''

#  打开训练损失和测试损失文件
train = open('Result/CNNResult/trainv20.pkl', 'rb')
trainloss = pkl.load(train)
test = open('Result/CNNResult/testv20.pkl', 'rb')
testloss = pkl.load(test)
'''testloss[93] = 5.97
testloss[76] = 6.8355'''
print('testloss:', testloss[99])
plt.figure()
xz = np.arange(0, len(trainloss) * 100, 100)
# 绘图，横轴起点为0，终点为len(loss_train)*100，步长为100
plt.plot(xz, trainloss, c='b', label="train dataset")
plt.plot(xz, testloss, c='r', label="test dataset")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

'''with open('Result/CNNResult/testv2.pkl', 'wb') as testv2:
    pkl.dump(testloss, testv2)  # 保存测试损失'''