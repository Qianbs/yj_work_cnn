import tensorflow as tf  
import sys  
import gzip  
import os  
import tempfile    
import numpy 
import time
import numpy as np
import socket
  
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
    print (time.strftime( ISOTIMEFORMAT, time.localtime() ),str)

#获取图片像素矩阵
def getImageFromFile(filename):       
    im = np.fromfile(filename,np.byte)
    im = im.reshape(1,784)
    return im

#开启一个tcp监听，接收web应用发送过来的图片，进行数字预测，并返回给web应用
def socket_server():
    HOST='127.0.0.1'
    PORT=50008
    s= socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
    s.bind((HOST,PORT)) 
    s.listen(1) 
    while 1:
        print ("wait for connect...")
        conn,addr=s.accept() 
        print ('Connected by',addr)
        conn.sendall(predict())
        conn.close()

#预测图片中的数字
def predict(): 
    #从约定的路径中读取待识别图片  
    image=getImageFromFile("/root/temp/imagedata")

    #加载图片到cnn神经网络计算，返回最大概率的分类，即为最大可能的数字
    printtime("predicit:")
    prediction=tf.argmax(y_conv,1)
    ret = sess.run([prediction,y_conv],feed_dict={x: image,keep_prob: 1.0})
    max = ret[0][0]
    print(max)
    return max

#######################################################################################################

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
init = tf.global_variables_initializer()
sess.run(init)

#加载模型参数文件
saver = tf.train.Saver()
saver.restore(sess, "mnist_variables/variables.ckpt")

#启动tcp监听，识别web应用发送过来的图片
socket_server()