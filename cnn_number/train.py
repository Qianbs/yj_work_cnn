import tensorflow as tf  
import sys  
import gzip  
import os  
import tempfile    
import numpy 
import time
from tensorflow.examples.tutorials.mnist import input_data  
  


def weight_variable(shape):  
  initial = tf.truncated_normal(shape, stddev=0.1)  
  return tf.Variable(initial)  
  
def bias_variable(shape):  
  initial = tf.constant(0.1, shape=shape)  
  return tf.Variable(initial)  
  
def conv2d(x, W):  
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  
  
def max_pool_2x2(x):  
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  


def printtime(str):
    ISOTIMEFORMAT='[%Y-%m-%d %X]'
    print(time.strftime( ISOTIMEFORMAT, time.localtime() ),str)

def currenttime():
    ISOTIMEFORMAT='[%Y-%m-%d %X]'
    return time.strftime( ISOTIMEFORMAT, time.localtime() )
#######################################################################################################

#模型参数路径 
variables_path = "mnist_variables/variables.ckpt"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  

#初始化tensorflow的session
sess = tf.InteractiveSession()  

#按照LeNet-5模型定义卷积神经网络的各层
x = tf.placeholder("float", shape=[None, 784])  
y_ = tf.placeholder("float", shape=[None, 10])  
  
W = tf.Variable(tf.zeros([784,10]))  
b = tf.Variable(tf.zeros([10]))  
  
#卷积层
W_conv1 = weight_variable([5, 5, 1, 32])  
b_conv1 = bias_variable([32])  
  
x_image = tf.reshape(x, [-1, 28, 28, 1])  
  
#池化层
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  
  
#卷积层
W_conv2 = weight_variable([5, 5, 32, 64])  
b_conv2 = bias_variable([64])  

#池化层 
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  
  
W_fc1 = weight_variable([7 * 7 * 64, 1024])  
b_fc1 = bias_variable([1024])  
  
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
  
keep_prob = tf.placeholder("float")  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
  
#全连接softmax层
W_fc2 = weight_variable([1024, 10])  
b_fc2 = bias_variable([10])  
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  
  
#定义tensorflow的计算  
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  
init = tf.global_variables_initializer()
sess.run(init)

#参数保存器
saver = tf.train.Saver()
  
#训练2万次
time_start = currenttime()
for i in range(20000):  
  batch = mnist.train.next_batch(50)  
  if i%100 == 0: 
    printtime(i)
    save_path = saver.save(sess, variables_path)
    print("Save to path: ", save_path)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  
  
#保存模型参数
save_path = saver.save(sess, variables_path)

#打印训练时间
print("Save to path: ", save_path)
printtime(i)
print("start:",time_start);
print("Training finished")
print("-------------------------------------------------")


#验证mnist测试集中的识别正确率
#batch = mnist.test.next_batch(1)
#print ("test accuracy %.3f" % accuracy.eval(feed_dict={  
#    x: batch[0], y_: batch[1], keep_prob: 1.0})) 





