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

import load
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
		    network, num_filters=32, filter_size=(3, 3),
		    nonlinearity=lasagne.nonlinearities.rectify,
		    W=lasagne.init.GlorotUniform(),pad=1, stride=1)

	network = lasagne.layers.Conv2DLayer(
                    network, num_filters=32, filter_size=(3, 3),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform(),pad=1, stride=1)
	

	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2,
		                     ignore_border=False)


	network = lasagne.layers.dropout(network, p=.25)


	network = lasagne.layers.Conv2DLayer(
		    network, num_filters=32, filter_size=(3, 3),
		    nonlinearity=lasagne.nonlinearities.rectify,
		    W=lasagne.init.GlorotUniform(),pad=1, stride=1)

	network = lasagne.layers.Conv2DLayer(
		        network, num_filters=32, filter_size=(3, 3),
		        nonlinearity=lasagne.nonlinearities.rectify,
		        W=lasagne.init.GlorotUniform(),pad=1, stride=1)
	

	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2,pad=(0,0), ignore_border=False)


	network = lasagne.layers.dropout(network, p=.25)


	network = lasagne.layers.Conv2DLayer(
		    network, num_filters=64, filter_size=(3, 3),
		    nonlinearity=lasagne.nonlinearities.rectify,
		    W=lasagne.init.GlorotUniform(),pad=1, untie_biases=True, stride=1)


	network = lasagne.layers.Conv2DLayer(
		    network, num_filters=64, filter_size=(3,3),
		    nonlinearity=lasagne.nonlinearities.rectify,
		    W=lasagne.init.GlorotUniform(), untie_biases=True, pad=1, stride=1)

	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2,pad=(0,0), ignore_border=False)

	network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.25),
            num_units=64,
            nonlinearity=lasagne.nonlinearities.rectify)

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
		biasnr = 0
		for param in cur_params:
			a = np.asarray(param.get_value())
			b = np.empty(a.shape)
			for index2, x in np.ndenumerate(a):
				if x>=0:
					#b[index2] = round((x * (1 << precision)) % (1 << bits))# * (1.0 / (1 << precision))
					if biasnr%2==0:
						b[index2] = np.trunc((x * (1 << precision)) % (1 << bits) + 0.5)# * (1.0 / (1 << precision))
					else:
						b[index2] = np.trunc((x * (1 << precision)) % (1 << bits) + 0.5) * (1 << precision)

				else:
					#b[index2] = -round((-x * (1 << precision)) % (1 << bits))# * (1.0/ (1 << precision))
					if biasnr%2==0:
						b[index2] = -np.trunc((-x * (1 << precision)) % (1 << bits) + 0.5)# * (1.0 / (1 << precision))
					else:
						b[index2] = -np.trunc((-x * (1 << precision)) % (1 << bits) + 0.5) * (1 << precision)

			lim_params.append(np.asarray(b, dtype=theano.config.floatX))
			biasnr += 1
		return lim_params


def lim_precision_inputs(inputs, bits, precision):
        lim_inputs = np.empty(inputs.shape)
        for index, value in np.ndenumerate(inputs):
                        if value>=0:
                                lim_inputs[index] = np.trunc((value * (1 << precision)) % (1 << bits) + 0.5)# * (1.0 / (1 << precision))
                        else:
                                lim_inputs[index] = -np.trunc((-value * (1 << precision)) % (1 << bits) + 0.5)# * (1.0 / (1 << precision))
        return lim_inputs.astype(theano.config.floatX)


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def resizeimages(imgs):
	resimg = []
	for img in imgs:
	#	resimg.append(rescale(img,24,24))
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
				output = output * (1.0 / (2 ** prec))
				outputneg = T.clip(output, -(2 ** (mod))+1, 0)
				outputpos = T.clip(output, 0, (2 ** (mod))-1)
				outputpos = T.switch(outputpos<(1.0 / (2 ** prec)), 0, outputpos)
				outputneg = T.switch(outputneg>-(1.0 / (2 ** prec)), 0, outputneg)
				output = T.floor((outputpos) % (2 ** (mod)) + 0.5) - T.floor((-outputneg) % (2 ** (mod)) + 0.5)
			if(layernr==13):
				debug = output
			layernr+=1
			all_outputs[layer] = output
	# return the output(s) of the requested layer(s) only
	layer = all_layers[-1]
	output = layer.get_output_for(all_outputs[layer.input_layer], deterministic = deterministic)
	output = output * (1.0 / (2 ** prec))
	outputneg = T.clip(output, -(2 ** (mod))+1, 0)
	outputpos = T.clip(output, 0, (2 ** (mod))-1)	
	outputpos = (T.floor(outputpos + 0.5) % (2 ** (mod)))# * (1.0 / (2 ** prec))						
	outputneg = -(-T.floor(outputneg + 0.5) % (2 ** (mod)))# * (1.0 / (2 ** prec))
	output = outputpos + outputneg
	softmax = lasagne.nonlinearities.softmax(output)# * (1.0 / (1 << prec))).eval()
	softmax = T.round(softmax * (2 ** bit)) * (1.0 / (2 ** bit)) #Use maximum precision for softmax	
	all_outputs[layer] = output#softmax
	try:
		return [all_outputs[layer] for layer in layer_or_layers]
	except TypeError:
		return all_outputs[layer_or_layers], debug #all_outputs[all_layers[4]] #layer_or_layers


def writeparams(param_values, network):
	nr = 0
	f = open('paramsmc_lim.csv', 'w')
	layers = lasagne.layers.get_all_layers(network)
	layernr = 1
	for params in param_values:
		while(isinstance(layers[layernr],lasagne.layers.DropoutLayer) or isinstance(layers[layernr],lasagne.layers.MaxPool2DLayer)):		#skipping dropoutlayers and poollayers
			layernr+=1
		if(nr%2==0)	:					#writing weights
			f.write('new layer\n')		#start new layer
			if(isinstance(layers[layernr],lasagne.layers.Conv2DLayer)):#extra params for convlayer
				f.write('conv,' + str(layers[layernr].stride[0]) + ',' + str(layers[layernr].pad[0]) + '\n')
				if(isinstance(layers[layernr+1],lasagne.layers.MaxPool2DLayer)):#if next layer is pool, write params
					f.write('Pool,')
					f.write(str(layers[layernr+1].pool_size[0]) + ',' + str(layers[layernr+1].stride[0]) + ',' + str(layers[layernr+1].pad[0]) + '\n')
			for dim in lasagne.layers.get_output_shape(layers[layernr-1])[1:]:	#write inputshape
				f.write(str(dim) + ',')
			f.write('\nw' + ',')			#weights
			for dim in params.shape:	#weightshape
				f.write(str(dim) + ',')
		else:						#writing bias
			f.write('b' + ',')			#bias
			for dim in params.shape:	#biasshape
				f.write(str(dim) + ',')
			layernr+=1	#go to next layer
		f.write('\nparams\n')#writing the values
		count=0
		for id, x in np.ndenumerate(params):
			f.write(str(x) + ',')
			count+=1
			if count>=10000:#this is not necessary, but gnumeric has limited columns
				f.write('\n')
				count = 0
		f.write('\nend params\n')
		nr+=1
	

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
# CNN = ?? weights
def main(train = False, train_lim = False, model='cifar',preproc = True, num_epochs=200,startepoch=199, bits = (16,), precision=((6,),), singleprec = False):
	dataloc = "Models/CIFAR10bis/"
	#dataloc = "/mnt/storage/users/ptimmerman/quantize/Models/CIFAR10/"
	# Load the dataset
	print("Loading data...")
	if preproc:
		    X_trainnoaug, y_trainnoaug, X_test, y_test = load.cifar10pre(dtype=theano.config.floatX, grayscale=False)
	else:
		    X_trainnoaug, y_trainnoaug, X_test, y_test = load.cifar10(dtype=theano.config.floatX, grayscale=False)


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
	bit_var = T.scalar('bits')
	prec_var = T.scalar('precision')
	epsilon = T.scalar()

	# Create neural network model (depending on first command line parameter)
	print("Building model and compiling functions...")
	if model == 'cifar':
		network = build_model(input_var)
		layers = lasagne.layers.get_all_layers(network)
		limnetwork = lasagne.layers.DenseLayer(
		    lasagne.layers.dropout(copy.deepcopy(layers[-3]), p=.5),
		    num_units=10,
		    nonlinearity=None)
		limlayers = lasagne.layers.get_all_layers(limnetwork)
	else:
		print("Unrecognized model type %r." % model)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_prediction = theano.printing.Print("pred: ")(test_prediction)
	test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_prediction, 0.00000001, 1.0-0.00000001),
		                                                    target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	predprint = theano.printing.Print("pred")(T.argmax(test_prediction, axis=1))
	test_acc = T.mean(T.eq(predprint, target_var),
		              dtype=theano.config.floatX)	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	
	test_prediction_lim,debug = get_output_lim(limnetwork, bit_var, prec_var, input_var, deterministic=True, singleprec = singleprec)
	test_prediction_lim = theano.printing.Print("pred: ")(test_prediction_lim)
	test_loss_lim = lasagne.objectives.categorical_crossentropy(T.clip(test_prediction_lim, epsilon,1.0-epsilon), target_var)
	test_loss_lim = test_loss_lim.mean()

	predprint_lim = theano.printing.Print("pred")(T.argmax(test_prediction_lim, axis=1))
	test_acc_lim = T.mean(T.eq(predprint_lim, target_var),
		              dtype=theano.config.floatX)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

	val_fn_lim = theano.function([input_var, target_var, bit_var, prec_var, epsilon], [test_loss_lim, test_acc_lim, debug], allow_input_downcast=True)

	totalerr = np.empty((num_epochs+1, np.array(precision).size+1))
	totalacc = np.empty((num_epochs+1, np.array(precision).size+1))

	print("Loading model...")
	with np.load('model9.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)
	lasagne.layers.set_all_param_values(limnetwork, param_values)

	bit = 7
	prec = 3
	eps = 1.0/(1<<prec)
#	writeparams(param_values, limnetwork)
	
	lim_params = lasagne.layers.get_all_params(limnetwork)
	lim_params = lim_precision_params(lim_params, bit, prec)
	print(lim_params[0])
	lasagne.layers.set_all_param_values(limnetwork, lim_params)
#	print("Saving model...")
#	np.savez('testmodelmc_lim.npz', *lasagne.layers.get_all_param_values(limnetwork))

	input = X_val[:100]
	target = y_val[:100]
#	val_err = 0
#	val_acc = 0
#	val_batches = 0
#	print("Starting validation...")
#	for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
#		    print("batchnr: ", val_batches+1)
#		    inputs, targets = batch
#		    err, acc = val_fn(inputs, targets)
#		    print(err, acc)
#		    val_err += err
#		    val_acc += acc
#		    val_batches += 1

	input = lim_precision_inputs(input, bit, prec)
#	print(input.shape)
#	f = open('inputmc_lim.csv', 'w')
#	for inp in input:	
#		for layer in inp:
#			for row in layer:
#				for elem in row:
#					f.write(str(elem) + ',')
#		f.write("\n")
#	print(target.shape)
#	f = open('targetmc_lim.csv', 'w')
#	for targets in target:
#		f.write(str(targets) + ",\n")
	
#	err, acc = val_fn(input, target)
#	print(input[0])
	err, acc, debug = val_fn_lim(input, target, bit, prec, eps)
	print("err: ", err, "--- acc: ", acc)
#	print(debug.shape)
#	for i in range(64):#
#		print(debug[0,i])
#	print(debug[0])
#	print("err: ", val_err/val_batches, "--- acc: ", val_acc/val_batches*100.0)


#	val_err = 0
#	val_acc = 0
#	val_batches = 0
#	print("Starting validation...")
#	for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
#		print("batchnr: ", val_batches+1)
#		inputs, targets = batch
#		err, acc = val_fn(inputs, targets)
#		val_err += err
#		val_acc += acc
#		print("err: ", err, "--- acc: ", acc)
#		val_batches += 1

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
