import numpy as np
import random

class Data:
    def __init__(self, num_train, num_test, n):
        self.A = self.getA(n)
        self.X_train, self.y_train = self.getData(num_train, n)
        self.X_test, self.y_test = self.getData(num_test, n)
        
        
    def getA(self,n):
        k = [np.exp(-5),np.exp(5)]
        A = [0]*n
        for i in range(n):
            if random.random()<0.5:
                A[i] = k[0]
            else:
                A[i] = k[1]
        return np.array(A)

    def U(self,q):
        A = np.diag(self.A)
        l = np.abs(q).max()
        if l<=3:
            return np.exp(-np.sqrt(np.dot(np.dot(q.T,A),q)))
        elif l<6:
            return np.exp(-np.sqrt(np.dot(np.dot(q.T,A),q))-1)
        else:
            return 0

    # k: number of samples
    def getData(self,k,n):
        data = np.zeros([k,n])
        label = np.zeros([k,1])
        for i in range(k):
            for j in range(n):
                data[i][j] = random.randint(-1000,1000)
        data = data/1000.0
        for i in range(k):
            label[i][0] = self.U(data[i])
        return data, label
    
