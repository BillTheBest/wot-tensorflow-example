#!/usr/bin/env python

"""
Generates a random stream of image data from MNIST data set

Each message is sent with the image data and the associated
label.  These messages can be read off the stream by the 
classifier.

"""

from wot import Wot
import numpy as np
import gzip

uint32 = np.dtype(np.uint32).newbyteorder('>')

def _zipfile(name):
	"""Returns a stream for the given gzip'd file"""
	return gzip.open(name)

def _read(stream):
	"""reads an int from the stream"""
	return np.frombuffer(stream.read(4), dtype=uint32)

def _image_header(stream):
	"""reads the headers from an MNIST image file"""
	magic = _read(stream)
	if magic != 2051:
		return (stream,0,0,0)
	images = _read(stream)
	rows = _read(stream)
	cols = _read(stream)
	return (stream,images,rows,cols)

def _image_data(stream,images,rows,cols):
	"""reads the data from an MNIST image file"""
	return np.frombuffer(stream.read(rows*cols*images), dtype=np.uint8).reshape(images,rows,cols,1)

def _label_header(stream):
	"""reads the headers from an MNIST label file"""
	magic = _read(stream)
	if magic != 2049:
		return (stream,labels)
	items = _read(stream)
	return (stream,items)

def _label_data(stream,items):
	"""reads the label data from an MNIST label file"""
	return np.frombuffer(stream.read(items), dtype=np.uint8)

def _one_hot(labels, classes):
	"""creates a "one hot" matrix for each label, ie puts a 1 at the index of the number"""
	num = labels.shape[0]
	offsets = np.arange(num) * classes
	one_hot = np.zeros((num,classes))
	one_hot.flat[ offsets + labels.ravel() ] = 1
	return one_hot	

def load(imagefile,labelfile):
	"""load the images and labels"""
	s = _zipfile(imagefile)
	(s,i,r,c) = _image_header(s)
	images = _image_data(s,i,r,c)
	s = _zipfile(labelfile)
	(s,l) = _label_header(s)
	labels = _label_data(s,l)
	return (images,labels)

def randomize(images,labels):
	"""randomizes the images, returns iterator"""
	indexes = np.arange(images.shape[0])
	np.random.shuffle(indexes)
	for i in indexes:
		yield (images[i],labels[i])	

def printlabel(label):
	print("sent label %d" % label)


def step(w,it):
	try:
		(image,label) = it.next()
		print("sending %d" % label)
		w.eval([ 
			(w.write_resource, ["mnist", np.ndarray.dumps(image), ("%d" % label) ]),
			(step, [w, it])
		])
	except StopIteration:
		print("done")
		w.eval([ (exit, [0]) ])

def iterate(w,images,labels):
	print("iterating")
	it = randomize(images,labels)
	step(w,it)


def stream(url,images,labels):
	"""generate a stream of messages"""
	w = Wot(url)	
	w.start([ 
		(w.new_channel, []),
		(iterate, [ w, images, labels ])
	])

def start(url,imagefile,labelfile):
	(images,labels) = load(imagefile,labelfile)
	stream(url,images,labels)

if __name__ == "__main__":
	start("amqp://test:test@localhost:5672/wot","MNIST_data/train-images.idx3-ubyte.gz","MNIST_data/train-labels.idx1-ubyte.gz")

