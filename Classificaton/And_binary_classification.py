#!usr/bin/env python
#coding=utf-8

import tensorflow as tf 
BATCHSIZE = 5000

#参数初始化

X_train = tf.placeholder(tf.float32,shape=[None,2])
Y_train = tf.placeholder(tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))

#sigmoid 假设函数

hypothesis = tf.sigmoid(tf.matmul(X_train,W)+b)

cost = -tf.reduce_mean(Y_train*tf.log(hypothesis)+(1-Y_train)*tf.log(1-hypothesis))
predicted = tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y_train),dtype=tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session()as sess:
	sess.run(tf.global_variables_initializer())
	print("before training W= {0},b={1}".format(sess.run(W),sess.run(b)))
	print("before training accuray = {0}".format(sess.run(accuracy,feed_dict={X_train:[[0,0],[0,1],[1,0],[1,1]],Y_train:[[0],[0],[0],[1]]})))
	for i in range(BATCHSIZE):
		sess.run(train_step,feed_dict={X_train:[[0,0],[0,1],[1,0],[1,1]],Y_train:[[0],[0],[0],[1]]})
	print("after training W= {0},b={1}".format(sess.run(W),sess.run(b)))
	
	print("after training accuray = {0}".format(sess.run(accuracy,feed_dict={X_train:[[0,0],[0,1],[1,0],[1,1]],Y_train:[[0],[0],[0],[1]]})))


