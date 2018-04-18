#!/usr/bin/env python
#coding=utf-8
BATCH_SIZE = 5000

import tensorflow as tf 
X_train = [[1,1],[1,0],[0,1],[1,2],[1,1.5]]
Y_train = [[14],[8],[9],[20],[17]]
W = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))
X_predict = [[0.0,0.0],[3.0,10.0],[-1.0,-2.0]]
#Y_predict = tf.matmul(X_predict,W)+b
W = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(X_train,W)+b
cost = tf.reduce_mean(tf.square(hypothesis-Y_train))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("before_train b is {0},w is {1}".format(sess.run(b),sess.run(W)))
	for i in range(BATCH_SIZE):
		sess.run(train_step)
		if i%500 == 0:
			print("after {0} training loss is {1}".format(i,sess.run(cost)))
 
	print("after_train b is {0},w is {1}".format(sess.run(b),sess.run(W)))
	print("y_predict is{0})".format(sess.run(tf.matmul(X_predict,W)+b)))

