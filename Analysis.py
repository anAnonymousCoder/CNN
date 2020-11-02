import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

'''
分析CNN估测的降水强度和实际降水强度的偏差
'''
est = open('Result/BPResult/estbp0.pkl', 'rb')
estimation = pkl.load(est)
ob = open('Result/BPResult/obbp0.pkl', 'rb')
observation = pkl.load(ob)

'''print(observation)
print(estimation)'''

plt.figure(figsize=(6, 6))
plt.scatter(estimation, observation, c='', s=10, marker='o', edgecolors='b')
plt.xlabel("estimation(mm/h)")
plt.ylabel("observation(mm/h)")
x = np.linspace(0, 60, 1000)
y = x
plt.plot(x, y, c='k', linestyle='--', label='y = x')
plt.title("The estimation of CNN model")
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.legend()
plt.show()

Number = []
for index in range(len(observation)):
    Number.append(index + 1)
Number = np.array(Number)

plt.figure(figsize=(10, 4))
plt.plot(Number, observation, c='b', label='observation', linewidth=1, alpha=0.75)
plt.plot(Number, estimation, c='r', linestyle='--', label='estimation', linewidth=1)
plt.xlabel('Sample Number')
plt.ylabel('precipitation intensity(mm/h)')
plt.legend()
plt.show()
'''def get_MSE(X, Y):  # 计算均方误差
    return sum([(x - y) ** 2 for x, y in zip(X, Y)]) / len(X)


print("MSE:", get_MSE(observation, estimation))'''
