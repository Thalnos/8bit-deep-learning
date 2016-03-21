#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
#import matplotlib.pyplot as plt
import copy

import load3
from PIL import Image
from PIL import ImageOps
import random
from random import shuffle

def update_progress(cur_val, end_val, bar_length=20):
    percent = float(cur_val) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\r[{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def build_model(input_var=None):

	network = lasagne.layers.InputLayer(shape=(None, 3, 24, 24),
		                                input_var=input_var)

	network = lasagne.layers.Conv2DLayer(
		    network, num_filters=64, filter_size=(3, 3),
		    nonlinearity=lasagne.nonlinearities.leaky_rectify,
		    W=lasagne.init.GlorotUniform(),pad=1, stride=1)

        network = lasagne.layers.Conv2DLayer(
                    network, num_filters=64, filter_size=(3, 3),
                    nonlinearity=lasagne.nonlinearities.leaky_rectify,
                    W=lasagne.init.GlorotUniform(),pad=1, stride=1)
	

	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2,
		                     ignore_border=False)


	network = lasagne.layers.dropout(network, p=.25)


	network = lasagne.layers.Conv2DLayer(
		    network, num_filters=128, filter_size=(3, 3),
		    nonlinearity=lasagne.nonlinearities.leaky_rectify,
		    W=lasagne.init.GlorotUniform(),pad=1, stride=1)

	network = lasagne.layers.Conv2DLayer(
		        network, num_filters=128, filter_size=(3, 3),
		        nonlinearity=lasagne.nonlinearities.leaky_rectify,
		        W=lasagne.init.GlorotUniform(),pad=1, stride=1)
	

	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2,pad=(0,0), ignore_border=False)


	network = lasagne.layers.dropout(network, p=.25)


	network = lasagne.layers.Conv2DLayer(
		    network, num_filters=256, filter_size=(3, 3),
		    nonlinearity=lasagne.nonlinearities.leaky_rectify,
		    W=lasagne.init.GlorotUniform(),pad=1, untie_biases=True, stride=1)


	network = lasagne.layers.Conv2DLayer(
		    network, num_filters=256, filter_size=(3,3),
		    nonlinearity=lasagne.nonlinearities.leaky_rectify,
		    W=lasagne.init.GlorotUniform(), untie_biases=True, pad=1, stride=1)

	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2,pad=(0,0), ignore_border=False)

	network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.25),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

	network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

	return network




# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def lim_precision_params(cur_params, bits, precision):
		lim_params = []
		for param in cur_params:
			a = np.asarray(param.get_value())
			b = np.empty(a.shape)
			for index2, x in np.ndenumerate(a):
				if x>=0:
					b[index2] = round((x * (1 << precision)) % (1 << bits)) * (1.0 / (1 << precision))
				else:
					b[index2] = -round((-x * (1 << precision)) % (1 << bits)) * (1.0/ (1 << precision))
			lim_params.append(np.asarray(b, dtype=theano.config.floatX))
		return lim_params

def lim_precision_inputs(inputs, bits, precision):
        lim_inputs = np.empty(inputs.shape)
        for index, value in np.ndenumerate(inputs):
                lim_inputs[index] = round((value * (1 << precision)) % (1 << bits)) * (1.0 / (1 << precision))
        return lim_inputs.astype(theano.config.floatX)

def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def generaternd(min, max):
	return (int((random.random()+0.5)*(2*max))%(2*max+1)+min, int((random.random()+0.5)*(2*max))%(2*max+1)+min)

def flipimage(image):  #(colorchannels, width, height)
	mirror = np.zeros(image.shape)
	for ch in range(image.shape[0]):
		width = image[ch,:,:].shape[0]
		height = image[ch,:,:].shape[1]
		for y in range(height):
		   for x in range(width // 2):
			   left = image[ch,y,x]
			   right = image[ch,y,width - 1 - x]
			   mirror[ch,y,width - 1 - x] = left
			   mirror[ch,y,x] = right
	return mirror

def showimages(images):
	for aug in images:
		aug2 = np.zeros((24,24,3))
		aug2[:,:,0] = aug[0,:,:]
		aug2[:,:,1] = aug[1,:,:]
		aug2[:,:,2] = aug[2,:,:]
		img2 = Image.fromarray(np.uint8(aug2*255.0))
		img2.show()	

def rescale(img, w,h):
	scale = np.zeros((32,32,3))
	scale[:,:,0] = img[0,:,:]
	scale[:,:,1] = img[1,:,:]
	scale[:,:,2] = img[2,:,:]
	res = np.asarray(Image.fromarray(np.uint8(scale*255.0)).resize((w,h), Image.ANTIALIAS))
	scaleimg = np.zeros((3,w,h))
	scaleimg[0,:,:] = np.asarray(res)[:,:,0]/255.0
	scaleimg[1,:,:] = np.asarray(res)[:,:,1]/255.0
	scaleimg[2,:,:] = np.asarray(res)[:,:,2]/255.0
	return scaleimg


def resizeimages(imgs):
	resimg = []
	for img in imgs:
#		resimg.append(rescale(img,24,24))
		resimg.append(img[:,4:-4,4:-4])
	return np.asarray(resimg)


#Reworked function from lasagne.layers.get_output(layer_or_layers, inputs=None, **kwargs)
#This version takes extra parameters bit, prec and deterministic
#Deterministic is used normally and enables or disables the dropoutlayers
#bit = #bits to represent a number
#prec = #bits of bit used to represent the fraction of the number
def get_output_lim(layer_or_layers, bit, prec, inputs=None, deterministic = False, singleprec = True,  **kwargs):
	from lasagne.layers import InputLayer
	from lasagne.layers import MergeLayer
	# obtain topological ordering of all layers the output layer(s) depend on
	treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
	all_layers = lasagne.layers.helper.get_all_layers(layer_or_layers, treat_as_input)
	# initialize layer-to-expression mapping from all input layers
	all_outputs = dict((layer, layer.input_var)
						for layer in all_layers
						if isinstance(layer, InputLayer) and
						layer not in treat_as_input)
	# update layer-to-expression mapping from given input(s), if any
	if isinstance(inputs, dict):
		all_outputs.update((layer, lasagne.utils.as_theano_expression(expr))
							for layer, expr in inputs.items())
	elif inputs is not None:
		if len(all_outputs) > 1:
			raise ValueError("get_output() was called with a single input "
		                     "expression on a network with multiple input "
		                     "layers. Please call it with a dictionary of "
		                     "input expressions instead.")
		for input_layer in all_outputs:
			all_outputs[input_layer] = lasagne.utils.as_theano_expression(inputs)
	# update layer-to-expression mapping by propagating the inputs
	layernr = 1
	mod = bit
	if singleprec:
		mod -= prec
	for layer in all_layers[:-1]:
		if layer not in all_outputs:
			try:
				if isinstance(layer, MergeLayer):
					layer_inputs = [all_outputs[input_layer]
								    for input_layer in layer.input_layers]
				else:
					layer_inputs = all_outputs[layer.input_layer]
			except KeyError:
				# one of the input_layer attributes must have been `None`
				raise ValueError("get_output() was called without giving an "
						         "input expression for the free-floating "
						         "layer %r. Please call it with a dictionary "
						         "mapping this layer to an input expression."
						         % layer)
			output = layer.get_output_for(layer_inputs, deterministic = deterministic)
			if (isinstance(layer,lasagne.layers.Conv2DLayer) or isinstance(layer,lasagne.layers.DenseLayer)) and not(layernr == len(all_layers)-1):
				output = T.switch(output<(1.0 / (2 ** prec)), 0, output)
				output = T.floor((output * (2 ** prec)) % (2 ** (mod)) + 0.5) * (1.0 / (2 ** prec))				
			layernr += 1
			all_outputs[layer] = output
	# return the output(s) of the requested layer(s) only
	layer = all_layers[-1]
	output = layer.get_output_for(all_outputs[layer.input_layer], deterministic = deterministic)
	output = output * (2 ** prec)
	outputneg = T.clip(output, -(2 ** (mod))+1, 0)
	outputpos = T.clip(output, 0, (2 ** (mod))-1)	
	outputpos = (T.floor(outputpos + 0.5) % (2 ** (mod))) * (1.0 / (2 ** prec))						
	outputneg = -(-T.floor(outputneg + 0.5) % (2 ** (mod))) * (1.0 / (2 ** prec))
	output = outputpos + outputneg
	softmax = lasagne.nonlinearities.softmax(output)# * (1.0 / (1 << prec))).eval()
	softmax = T.round(softmax * (2 ** bit)) * (1.0 / (2 ** bit)) #Use maximum precision for softmax	
	all_outputs[layer] = softmax

	try:
		return [all_outputs[layer] for layer in layer_or_layers]
	except TypeError:
		return all_outputs[layer_or_layers] #all_outputs[all_layers[4]] #layer_or_layers


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
# CNN = ?? weights
def main(train = False, train_lim = False, model='cifar', preproc = True, num_epochs=200,startepoch=199, bits = (), precision=(()), singleprec = True):
	#dataloc = "Models/CIFAR10tris/"
	dataloc = "/mnt/storage/users/ptimmerman/quantize/Models/CIFAR10quadbisnew/"
	# Load the dataset
	print("Loading data...")
	if preproc:
		X_trainnoaug, y_trainnoaug, X_test, y_test = load3.cifar10pre(dtype=theano.config.floatX, grayscale=False)
	else:
		X_trainnoaug, y_trainnoaug, X_test, y_test = load3.cifar10(dtype=theano.config.floatX, grayscale=False)

	# Extracting validationsets
	X_trainnoaug, X_val = X_trainnoaug[:-10000], X_trainnoaug[-10000:]
	y_trainnoaug, y_val = y_trainnoaug[:-10000], y_trainnoaug[-10000:]

	# Reshape data
	X_trainnoaug = X_trainnoaug.reshape((-1, 3, 32, 32))
	X_test = X_test.reshape((-1, 3, 32, 32))
	X_val = X_val.reshape((-1, 3, 32, 32))

	X_val = resizeimages(X_val)
	X_test = resizeimages(X_test)


	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')
	prediction_var = T.fmatrix('predictions')
	epsilon = T.scalar()

	# Create neural network model (depending on first command line parameter)
	print("Building model and compiling functions...")
	if model == 'cifar':
		network = build_model(input_var)
		layers = lasagne.layers.get_all_layers(network)
		limnetwork = lasagne.layers.DenseLayer(
		    lasagne.layers.dropout(layers[-3], p=.5),
		    num_units=10,
		    nonlinearity=None)
		limlayers = lasagne.layers.get_all_layers(limnetwork)
	else:
		print("Unrecognized model type %r." % model)

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction, 0.00000001, 1.0-0.00000001), target_var)
	loss = loss.mean()

	acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
		              dtype=theano.config.floatX)
	# We could add some weight decay as well here, see lasagne.regularization.

	#Loss using a forward pass with limited precision
	prediction_lim = get_output_lim(limnetwork, bit_var, prec_var, input_var, deterministic=False, singleprec = singleprec)
	loss_lim = lasagne.objectives.categorical_crossentropy(T.clip(prediction_lim, epsilon,1.0-epsilon), target_var)
	loss_lim = loss_lim.mean()

	acc_lim = T.mean(T.eq(T.argmax(prediction_lim, axis=1), target_var),
		              dtype=theano.config.floatX)
	

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
		    loss, params, learning_rate=0.01, momentum=0.9)

	params_lim = lasagne.layers.get_all_params(limnetwork, trainable=True)
	updates_lim = lasagne.updates.nesterov_momentum(loss_lim, params_lim, learning_rate=0.01, momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_prediction, 0.00000001, 1.0-0.00000001),
		                                                    target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:

	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
		              dtype=theano.config.floatX)	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction, 0.00000001, 1.0-0.00000001), target_var)
	loss = loss.mean()

	acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
		              dtype=theano.config.floatX)
	# We could add some weight decay as well here, see lasagne.regularization.

	#Loss using a forward pass with limited precision
	prediction_lim = get_output_lim(limnetwork, bit_var, prec_var, input_var, deterministic=False, singleprec = singleprec)
	loss_lim = lasagne.objectives.categorical_crossentropy(T.clip(prediction_lim, epsilon,1.0-epsilon), target_var)
	loss_lim = loss_lim.mean()

	acc_lim = T.mean(T.eq(T.argmax(prediction_lim, axis=1), target_var),
		              dtype=theano.config.floatX)
	

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
		    loss, params, learning_rate=0.01, momentum=0.9)

	params_lim = lasagne.layers.get_all_params(limnetwork, trainable=True)
	updates_lim = lasagne.updates.nesterov_momentum(loss_lim, params_lim, learning_rate=0.01, momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass

	test_prediction_lim = get_output_lim(limnetwork, bit_var, prec_var, input_var, deterministic=True, singleprec = singleprec)
	test_loss_lim = lasagne.objectives.categorical_crossentropy(T.clip(test_prediction_lim, epsilon,1.0-epsilon), target_var)
	test_loss_lim = test_loss_lim.mean()

	test_acc_lim = T.mean(T.eq(T.argmax(test_prediction_lim, axis=1), target_var),
		              dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], [loss, acc], updates=updates, allow_input_downcast=True)

	train_fn_lim = theano.function([input_var, target_var, bit_var, prec_var, epsilon], [loss_lim, acc_lim], updates=updates_lim, allow_input_downcast=True)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

	val_fn_lim = theano.function([input_var, target_var, bit_var, prec_var, epsilon], [test_loss_lim, test_acc_lim], allow_input_downcast=True)

	totalerr = np.empty((num_epochs+1, np.array(precision).size+1))
	totalacc = np.empty((num_epochs+1, np.array(precision).size+1))

	if startepoch>0:
		with np.load(dataloc + 'model' + str(startepoch-1) + '.npz') as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(network, param_values)

	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(startepoch,num_epochs):
		start_time = time.time()
		if train:
			randpattern = range(0,16)
			shuffle(randpattern)
			train_err = 0
			train_acc = 0
			train_batches = 0  
			for i in randpattern:
				print("batch: ", i/4, ", subbatch: ", i%4)
				X_train, y_train = load3.cifar10augpre(batch=i/4 , subbatch=i%4)
				# In each epoch, we do a full pass over the training data:
				for batch in iterate_minibatches(X_train, y_train, 64, shuffle=True):
					update_progress(train_batches, X_train.shape[0]/64)
					inputs, targets = batch
					err, acc = train_fn(inputs, targets)
					train_err += err
					train_acc += acc				
					train_batches += 1
				print("")
			np.savez(dataloc + 'model' + str(epoch) + '.npz', *lasagne.layers.get_all_param_values(network))
		elif train_lim:#TODO change for augmented data
			print("Augmenting data...")
			bit = bits[0]
			prec = precision[0][0]
			print(bit, prec)
			eps = (1.0 / (1 << bit))
			X_train, y_train = augmentdata(X_trainnoaug, y_trainnoaug)
			X_train_lim = lim_precision_inputs(X_train, bit, prec, eps)
			# In each epoch, we do a full pass over the training data:
			train_err_lim = 0
			train_acc_lim = 0
			train_batches = 0
			for batch in iterate_minibatches(X_train_lim, y_train, 500, shuffle=True):
				print("batch: ", train_batches + 1)
				inputs, targets = batch
				current_params = lasagne.layers.get_all_params(network)
				lim_params = lim_precision_params(current_params, bit, prec)
				lasagne.layers.set_all_param_values(limnetwork, lim_params)
				err, acc = train_fn_lim(inputs, targets, bit, prec)
				train_err_lim += err
				train_acc_lim += acc
				train_batches += 1
			np.savez(dataloc + 'model_lim' + str(bit) + '--' + str(prec) + '--' + str(epoch) + '.npz', *lasagne.layers.get_all_param_values(network))		
		else:
			with np.load(dataloc + 'model' + str(epoch) + '.npz') as f:
				param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			lasagne.layers.set_all_param_values(network, param_values)
		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		limval_err = np.zeros(np.array(precision).shape)
		limval_acc = np.zeros(np.array(precision).shape)

		
		current_params = copy.deepcopy(lasagne.layers.get_all_params(network))
		
	
		old_params = lasagne.layers.get_all_params(network)
		for index, param in np.ndenumerate(current_params):
		     old_params[index[0]] = np.asarray(current_params[index[0]].get_value(), dtype=theano.config.floatX)
		print("Starting validation...")
		for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
			print("batchnr: ", val_batches+1)
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1
		for bit,precisions in zip(bits,precision):
			for prec in precisions:
				print(bit, prec)
				lim_params = lim_precision_params(current_params, bit, prec)
				lasagne.layers.set_all_param_values(limnetwork, lim_params)
				for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
					eps = (1.0 / (1 << bit))
					inputs, targets = batch
					lim_inputs = lim_precision_inputs(inputs, bit, prec)
					err, acc = val_fn_lim(lim_inputs, targets, bit, prec, eps)
					print(err, acc)
					limval_err[bits.index(bit)][precisions.index(prec)] += err
					limval_acc[bits.index(bit)][precisions.index(prec)] += acc
		lasagne.layers.set_all_param_values(network, old_params)
		    

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
		    epoch + 1, num_epochs, time.time() - start_time))
		if train:
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  training loss:\t\t{:.6f}".format(train_acc / train_batches * 100))
		elif train_lim:
			print("  training loss:\t\t{:.6f}".format(train_err_lim / train_batches))
			print("  training loss:\t\t{:.6f}".format(train_acc_lim / train_batches * 100))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

		for bit,precisions in zip(bits,precision):
			for prec in precisions:
				print(str(bit) + " bits precision, ", str(prec), " bits after point")
				print("  limval loss:\t\t{:.6f}".format(limval_err[bits.index(bit)][precisions.index(prec)] / val_batches))
				print("  limval accuracy:\t\t{:.2f} %".format(
						limval_acc[bits.index(bit)][precisions.index(prec)] / val_batches * 100))
		concaterr = []	        
		concatacc = []
		for err, acc in zip(limval_err[:], limval_acc[:]):
			concaterr = np.concatenate((concaterr, err[:] / val_batches))
			concatacc = np.concatenate((concatacc, acc[:] / val_batches * 100))
		concaterr = np.concatenate((concaterr, [val_err / val_batches]))
		concatacc = np.concatenate((concatacc, [val_acc / val_batches * 100]))	
		totalerr[epoch] = concaterr
		totalacc[epoch] = concatacc

		np.savetxt(dataloc + str(bits) + str(precision) + "err.csv", totalerr, delimiter=",")
		np.savetxt(dataloc + str(bits) + str(precision) + "acc.csv", totalacc, delimiter=",")  

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	limtest_err = np.zeros(np.array(precision).shape)
	limtest_acc = np.zeros(np.array(precision).shape)

	if train:
		np.savez(dataloc + 'finalmodel.npz', *lasagne.layers.get_all_param_values(network))
	else:
		with np.load(dataloc + 'finalmodel.npz') as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(network, param_values)
	current_params = copy.deepcopy(lasagne.layers.get_all_params(network))
	old_params = lasagne.layers.get_all_params(network)
	for index, param in np.ndenumerate(current_params):
		 old_params[index[0]] = np.asarray(current_params[index[0]].get_value(), dtype=theano.config.floatX)
	for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
		print("batchnr: ", test_batches+1)
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
	for bit, precisions in zip(bits,precision):
		for prec in precisions:
			print(bit, prec)	
			lim_params = lim_precision_params(current_params, bit, prec)
			lasagne.layers.set_all_param_values(limnetwork, lim_params)
			for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
				eps = (1.0 / (1 << bit))
				inputs, targets = batch
				lim_inputs = lim_precision_inputs(inputs, bit, prec)
				err, acc = val_fn_lim(lim_inputs, targets, bit, prec, eps)
				print(err, acc)
				limtest_err[bits.index(bit)][precisions.index(prec)] += err
				limtest_acc[bits.index(bit)][precisions.index(prec)] += acc
				
		

	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))
	for bit, precisions in zip(bits,precision):
		for prec in precisions:
			print(str(bit) + " bits precision, ", str(prec), " bits after point")
			print("  limtest loss:\t\t\t{:.6f}".format(limtest_err[bits.index(bit)][precisions.index(prec)] / test_batches))
			print("  limtest accuracy:\t\t{:.2f} %".format(
			   limtest_acc[bits.index(bit)][precisions.index(prec)] / test_batches * 100))

	concaterr = []	        
	concatacc = []
	for err, acc in zip(limtest_err[:], limtest_acc[:]):
		concaterr = np.concatenate((concaterr, err[:] / test_batches))
		concatacc = np.concatenate((concatacc, acc[:] / test_batches * 100))
	concaterr = np.concatenate((concaterr, [test_err / test_batches]))
	concatacc = np.concatenate((concatacc, [test_acc / test_batches * 100]))	
	totalerr[-1] = concaterr
	totalacc[-1] = concatacc    

	np.savetxt(dataloc + str(bits) + str(precision) + "err.csv", totalerr, delimiter=",")
	np.savetxt(dataloc + str(bits) + str(precision) + "acc.csv", totalacc, delimiter=",")
	# Optionally, you could now dump the network weights to a file like this:
	# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
	#
	# And load them again later on like this:
	# with np.load('model.npz') as f:
	#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# lasagne.layers.set_all_param_values(network, param_values)
	x = np.arange(0,num_epochs-startepoch+1, 1)
	plt.figure(1)
	ax = plt.subplot(211)

	for errtot in np.transpose(totalerr[startepoch:,:])[:-1]:
		plt.plot(x, errtot)
	plt.plot(x, np.transpose(totalerr[startepoch:,:])[-1])
	plt.xlabel('epochs', fontsize=18)
	plt.ylabel('loss', fontsize=16)
	ax.set_yscale('log')

	plt.subplot(212)

	i = 0
	j = 0
	print(totalacc.shape)
	print(totalacc[startepoch:,:].shape)
	for acctot in np.transpose(totalacc[startepoch:,:])[:-1]:
		name = str(bits[i]) + " total bits, " + str(precision[i][j]) + " fraction bits"	
		plt.plot(x, acctot, label=name)
		j += 1
		if j == len(precision[i]):
			j = 0
			i += 1
	plt.plot(x, np.transpose(totalacc[startepoch:,:])[-1], label='float32')
	plt.xlabel('epochs', fontsize=18)
	plt.ylabel('accuracy', fontsize=16)
	plt.legend(bbox_to_anchor=(0.6, 0.3), loc=2, borderaxespad=0.)

	plt.show()

if __name__ == '__main__':
	if ('--help' in sys.argv) or ('-h' in sys.argv):
		print("Trains a neural network on MNIST using Lasagne.")
		print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
		print()
		print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
		print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
		print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
		print("       input dropout and DROP_HID hidden dropout,")
		print("       'cnn' for a simple Convolutional Neural Network (CNN).")
		print("EPOCHS: number of training epochs to perform (default: 500)")
	else:
		kwargs = {}
		if len(sys.argv) > 1:
		    kwargs['model'] = sys.argv[1]
		if len(sys.argv) > 2:
		    kwargs['num_epochs'] = int(sys.argv[2])
		if len(sys.argv) > 3:
		    kwargs['precision'] = int(sys.argv[3])
		main(**kwargs)
