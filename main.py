import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

class Data:
    def __init__(self,num_train,num_test):
        self.A = np.exp(-5)
        self.X_train, self.y_train = self.getData(num_train)
        self.X_test, self.y_test = self.getData(num_test)

    def U(self,q):
        if np.abs(q) <= 3:
            return np.exp(-np.sqrt(q*self.A*q))
        elif np.abs(q) < 6:
            return np.exp(-np.sqrt(q*self.A*q)-1)
        else:
            return 0

    def getData(self,k):
        data = np.zeros(k)
        label = np.zeros(k)
        for i in range(k):
            data[i] = random.randint(-1000,1000)
        data = data/100.0
        for i in range(k):
            label[i] = self.U(data[i])
        return data, label

n = 1
num_train = 10000
num_test = 2000
data = Data(num_train, num_test)

plt.scatter(data.X_train, data.y_train)
plt.show()

