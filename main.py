import tensorflow as tf
import numpy as np
from random import random,randint
from data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 2
num_train = 10000
num_test = 2000
data = Data(num_train, num_test, n)       
num_epoch = 500
batch = 10
num_hidden = 10

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def eculidean_distance(x,y):
    return np.linalg.norm(x-y)

def C_initialize(num_hidden,n):
    return np.random.rand(num_hidden,n)

def sigma_initialize(C):
    num_hidden = C.shape[0]
    sigma = np.zeros(num_hidden)
    for i in range(num_hidden):
        for j in range(num_hidden):
            delta = C[i] - C[j]
            sigma[i] += np.dot(delta, delta.T)
    sigma /= num_hidden
    return sigma

def generate_input(data,C):
    sigma = sigma_initialize(C)
    K = np.zeros([data.shape[0],C.shape[0]])
    for i in range(data.shape[0]):
        for j in range(C.shape[0]):
            delta = data[i:i+1] - C[j:j+1]
            K[i][j] = np.dot(delta,delta.T)/(2*sigma[j])
    K = np.exp(-K)
    return K

C = C_initialize(num_hidden,n)
sigma = sigma_initialize(C)
X_train = generate_input(data.X_train, C)
X_test = generate_input(data.X_test, C)

print 'initial dataset:'
print 'training data',data.X_train.shape,'training label',data.y_train.shape
print 'testing data',data.X_test.shape,'testing label',data.y_test.shape
print

print 'C matrix:'
print C.shape
print

print 'sigma matrix:'
print sigma.shape
print

print 'preprocessed input:'
print 'training input',X_train.shape
print 'testing input',X_test.shape
print 

print 'training dataset 0:',X_train[0]
print 'training dataset 1:',X_train[1]
print 'training dataset 2:',X_train[2]
print

X = tf.placeholder(tf.float32,[None,num_hidden])
y_ = tf.placeholder(tf.float32,[None,1])

W = weight_variable([num_hidden,1])
b = bias_variable([1])

# W = tf.clip_by_value(W,0,1)

y = tf.matmul(X,W)+b

loss = tf.reduce_mean(tf.square(y - y_))
opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# opt = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9, momentum=0.99, epsilon=1e-8, use_locking=False, name='RMSProp').minimize(loss)

init = tf.global_variables_initializer()
last_l = 0.0
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epoch):
        for j in range(num_train/batch):
            data_batch = X_train[i*batch:(i+1)*batch]
            label_batch = data.y_train[i*batch:(i+1)*batch]
            sess.run(opt, feed_dict={X:data_batch,y_:label_batch})
        l = sess.run(loss, feed_dict={X:X_test,y_:data.y_test})
        print l,
        if abs(last_l - l) < 1e-4:
            break
        last_l = l
    p = sess.run(y, feed_dict={X:X_test, y_:data.y_test})
    print
    print sess.run(b)
    print sess.run(W)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x = [data.X_train[i,0] for i in range(num_train)]
y = [data.X_train[i,1] for i in range(num_train)]
z = [data.y_train[i,0] for i in range(num_train)]
ax.scatter(x,y,z,c='b')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x = [data.X_test[i][0] for i in range(num_test)]
y = [data.X_test[i][1] for i in range(num_test)]
z = [p[i][0] for i in range(num_test)]
ax.scatter(x,y,z,c='r')
plt.show()










