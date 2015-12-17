import tensorflow as tf
import numpy as np
from wot import Wot
import sys
import os

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

iterations = 0

images = []
labels = []

def train(message,meta,headers):
	img = np.loads(message)
	img = img.reshape(1, 28 * 28)
	img = img.astype(np.float32)
	img = np.multiply(img, 1.0 / 255.0)

	label = np.array([np.eye(10)[1*meta]])
	images.append(img[0])
	labels.append(label[0])	
	sess.run(train_step, feed_dict={x: img, y_: label})
	++iterations
	if (iterations % 100) == 0:
		print(sess.run(accuracy, feed_dict={x: images, y_: labels}))

w = Wot("amqp://test:test@127.0.0.1:5672/wot")
w.start( [ 
	( w.new_channel, []),
	( w.stream_resource, [ "mnist", train ]) 
])
