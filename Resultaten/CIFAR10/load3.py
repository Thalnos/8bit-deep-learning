import numpy as np
import os
import cPickle as pickle
import glob
from scipy.misc import imread
import random
from pylearn2.datasets.zca_dataset import ZCA_Dataset    
from pylearn2.utils import serial


data_dir = "/mnt/storage/users/ptimmerman/iCub/data/data"
#data_dir = "data/"
data_dir_cifar10 = os.path.join(data_dir, "cifar-10-batches-py")
data_dir_cifar10pre = os.path.join(data_dir, "cifar10/pylearn2_gcn_whitened")
data_dir_cifar10preaug = os.path.join(data_dir, "cifar10/augmentpreproc")
#data_dir_cifar100 = os.path.join(data_dir, "cifar-100-python")
#data_dir_icub28 = os.path.join(data_dir, "iCubWorld28")

#class_names_cifar10 = np.load(os.path.join(data_dir_cifar10, "batches.meta"))
#class_names_cifar100 = np.load(os.path.join(data_dir_cifar100, "meta"))


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch_cifar10(filename, dtype='float64'):
    """
    load a batch in the CIFAR-10 format
    """
    path = os.path.join(data_dir_cifar10, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0 # scale between [0, 1]
    labels = one_hot(batch['labels'], n=10) # convert labels to one-hot representation
    return data.astype(dtype), labels.astype(dtype)

def _load_batch_cifar10pre(dtype='float64'):
	"""
	load a batch in the CIFAR-10 format
	"""
	preproc = os.path.join(data_dir_cifar10pre, "preprocessor.pkl")
	preprocessor = serial.load(preproc)
	train = os.path.join(data_dir_cifar10pre, "train.pkl")
	train_set = ZCA_Dataset(preprocessed_dataset=serial.load(train), preprocessor = preprocessor, start=0, stop = 50000)
	test = os.path.join(data_dir_cifar10pre, "test.pkl")
	test_set = ZCA_Dataset(preprocessed_dataset= serial.load(test), preprocessor = preprocessor)

	return train_set, test_set



def _grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


def cifar10(dtype='float64', grayscale=True):
    # train
    x_train = []
    t_train = []
    for k in xrange(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)	
    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)
    # test
    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)
    return x_train, np.argmax(t_train,axis=1), x_test, np.argmax(t_test,axis=1)

def cifar10pre(dtype='float64', grayscale=True):
	# train
	train, test = _load_batch_cifar10pre()
	x_train = train.X
	t_train = train.y		

	t_train = np.concatenate(t_train, axis=0)

	# test
	x_test = test.X
	t_test = test.y

	t_test = np.concatenate(t_test, axis=0)

	if grayscale:
		x_train = _grayscale(x_train)
		x_test = _grayscale(x_test)
	return x_train, t_train, x_test, t_test

def cifar10augpre(batch , subbatch):
	inputs = np.load(data_dir_cifar10preaug + "aug" + str(subbatch) + "input" + str(batch) + ".npz")
	targets = np.load(data_dir_cifar10preaug + "aug" + str(subbatch) + "target" + str(batch) + ".npz")
	return inputs, targets

def icub_load(filename, dtype='float64', batches=1, types=1):
	print filename
	dir_icub = os.path.join(data_dir_icub28, filename)
	data = []
	labels = []
	classes = []
	for batchnr in range(batches):
		print "day%d" % (batchnr + 1)
		daydir = os.path.join(dir_icub, "day%d" % (batchnr + 1))
		for objecttype in os.listdir(daydir):
			print "type: ", objecttype
			if objecttype not in classes:
				classes.append(objecttype)
			for typenr in range(types):
				imagesdir = os.path.join(daydir, objecttype, ''.join((objecttype,"%d" % (typenr + 1))))
				for imagefiles in os.listdir(imagesdir):
					im = imread(os.path.join(imagesdir, imagefiles)) / 255.0
					if im.shape[0]==128 and im.shape[1]==128:
						data.append(im.astype(dtype))
						labels.append(classes.index(objecttype))
	return data, labels, classes

def icub(dtype='float64', batches=1, types=1):
	x_train, t_train, classes = icub_load("train", dtype, batches, types)

	combined = zip(t_train, x_train)
	random.shuffle(combined)
	t_train[:], x_train[:] = zip(*combined)
	t_train = one_hot(t_train, n=len(classes))

	x_test, t_test, classes = icub_load("test", dtype, batches, types)
	combined = zip(t_test, x_test)
	random.shuffle(combined)
	t_test[:], x_test[:] = zip(*combined)
	t_test = one_hot(t_test, n=len(classes))
	return np.array(x_train), t_train.astype(dtype), np.array(x_test), t_test.astype(dtype)
		

def _load_batch_cifar100(filename, dtype='float64'):
    """
    load a batch in the CIFAR-100 format
    """
    path = os.path.join(data_dir_cifar100, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0
    labels = one_hot(batch['fine_labels'], n=100)
    return data.astype(dtype), labels.astype(dtype)


def cifar100(dtype='float64', grayscale=True):
    x_train, t_train = _load_batch_cifar100("train", dtype=dtype)
    x_test, t_test = _load_batch_cifar100("test", dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)

    return x_train, t_train, x_test, t_test
