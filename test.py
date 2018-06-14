import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def generate_input(input, C):
    output = np.zeros([input.shape[0], C.shape[0]])
    for i in range(input.shape[0]):
        for j in range(C.shape[0]):
            output[i,j] = np.exp(-np.square(input[i] - C[j])/2)
    return output

num_train = 10000
num_test = 2000
num_epoch = 11
batch = 10
num_hidden = 50
num_output = 1

input_train = (np.random.random([num_train,1]) - 0.5)*10
C = (np.random.random(num_hidden) - 0.5)*10
X_train = generate_input(input_train, C)
y_train = 0.5*np.exp(-np.square(input_train)/2) + 0.3*np.exp(-np.square(input_train - 1)/4) + 0.2*np.exp(-np.square(input_train - 3)/8)

print(X_train.shape, y_train.shape)

input_test = np.sort((np.random.random([num_test,1]) - 0.5)*10, axis=0)
X_test = generate_input(input_test, C)
y_test = 0.5*np.exp(-np.square(input_test)/2) + 0.3*np.exp(-np.square(input_test - 1)/4) + 0.2*np.exp(-np.square(input_test - 3)/8)

print(X_test.shape, y_test.shape)


X = tf.placeholder(tf.float32, shape=[None, num_hidden])
y_ = tf.placeholder(tf.float32, shape=[None, num_output])

W = weight_variable([num_hidden, num_output])
b = bias_variable([num_output])

y = tf.matmul(X, W) + b

loss = tf.reduce_mean(tf.square(y - y_))
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    plt.subplot(3,4,12)
    plt.plot(input_test, y_test)
    for _ in range(num_epoch):
        for i in range(num_train/batch):
            start = i*batch
            end = (i+1)*batch
            image = X_train[start:end]
            label = y_train[start:end]
            l, tr = sess.run([loss, opt], feed_dict={X: image, y_: label})
            if i%100 == 0:
                print "step =", i, "loss =", l
        l, yy = sess.run([loss, y], feed_dict={X: X_test, y_: y_test})
        print(l)
        plt.subplot(3,4,_+1)
        plt.plot(input_test, yy)
    plt.show()


