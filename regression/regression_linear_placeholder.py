#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf 
BATCH_SIZE = 5000
#初始化数数据
X_train = tf.placeholder(tf.float32,shape=[None,2])
Y_train = tf.placeholder(tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))
#待预测数据
X_predict = tf.placeholder(tf.float32,shape=[None,2])
#假设函数
hypothesis = tf.matmul(X_train,W)+b
#损失函数
cost = tf.reduce_mean(tf.square(hypothesis-Y_train))
#梯度下降法求解参数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("before_train b is {0},w is {1}".format(sess.run(b),sess.run(W)))
	for i in range(BATCH_SIZE):
		sess.run(train_step,feed_dict={X_train:[[1,1],[1,0],[0,1],[1,2],[1,1.5]],Y_train:[[14],[8],[9],[20],[17]]})
		if i%500 == 0:
			print("after {0} training loss is {1}".format(i,sess.run(cost,feed_dict={X_train:[[1,1],[1,0],[0,1],[1,2],[1,1.5]],Y_train:[[14],[8],[9],[20],[17]]})))
 
	print("after_train b is {0},w is {1}".format(sess.run(b),sess.run(W)))
	print("y_predict is{0})".format(sess.run(tf.matmul(X_predict,W)+b,feed_dict={X_predict:[[0.0,0.0],[1.0,10.0]]})))

