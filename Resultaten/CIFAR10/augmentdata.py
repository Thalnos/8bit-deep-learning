#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import matplotlib.pyplot as plt
import copy

import load3
from PIL import Image
from PIL import ImageOps
import random
from random import shuffle

data_dir = "data/cifar10/augmentpreproc/"

def update_progress(cur_val, end_val, bar_length=20):
    percent = float(cur_val) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\r[{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

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

def rescale28(img, w,h):
	scale = np.zeros((28,28,3))
	scale[:,:,0] = img[0,:,:]
	scale[:,:,1] = img[1,:,:]
	scale[:,:,2] = img[2,:,:]
	res = np.asarray(Image.fromarray(np.uint8(scale*255.0)).resize((w,h), Image.ANTIALIAS))
	scaleimg = np.zeros((3,w,h))
	scaleimg[0,:,:] = np.asarray(res)[:,:,0]/255.0
	scaleimg[1,:,:] = np.asarray(res)[:,:,1]/255.0
	scaleimg[2,:,:] = np.asarray(res)[:,:,2]/255.0
	return scaleimg

def augmenttranslate(img, min, max):
	xy = generaternd(min,max)
	if xy[0] == min:
		augimg = img[:,max-xy[0]:,:]
	else:
		augimg = img[:,max-xy[0]:min-xy[0],:]
	if xy[1] == min:
		augimg = augimg[:,:,max-xy[1]:]
	else:
		augimg = augimg[:,:,max-xy[1]:min-xy[1]]
	return augimg

def augmentdata(inputs, targets, batchnr):
	augmentedinputs0 = []
	augmentedtargets0 = []
	augmentedinputs1 = []
	augmentedtargets1 = []
	augmentedinputs2 = []
	augmentedtargets2 = []
	augmentedinputs3 = []
	augmentedtargets3 = []

	rndnr = 0
	imgnr = 0
	for img, target in zip(inputs[:], targets[:]):
		update_progress(imgnr, inputs.shape[0])
		rand = []
		for i in range(0,4):
			for j in range(0,17):
				rand.append(i)
		shuffle(rand)
		
		x = np.arange(-4,5,2)
		y = np.arange(-4,5,2)

		for stepx in x:
			for stepy in y:
				if stepy == 4:
				        augimg = img[:,:,4+stepy:]
				else:
				        augimg = img[:,:,4+stepy:-4+stepy]
				if stepx == 4:
				        augimg = augimg[:,4+stepx:,:]
				else:
				        augimg = augimg[:,4+stepx:-4+stepx,:]

				if rand[rndnr]==0:
					augmentedinputs0.append(augimg)
					augmentedtargets0.append(target)
				elif rand[rndnr]==1:
					augmentedinputs1.append(augimg)
					augmentedtargets1.append(target)
				elif rand[rndnr]==2:
					augmentedinputs2.append(augimg)
					augmentedtargets2.append(target)
				elif rand[rndnr]==3:
					augmentedinputs3.append(augimg)
					augmentedtargets3.append(target)
				rndnr+=1
				if rand[rndnr]==0:
					augmentedinputs0.append(flipimage(augimg))
					augmentedtargets0.append(target)
				elif rand[rndnr]==1:
					augmentedinputs1.append(flipimage(augimg))
					augmentedtargets1.append(target)
				elif rand[rndnr]==2:
					augmentedinputs2.append(flipimage(augimg))
					augmentedtargets2.append(target)
				elif rand[rndnr]==3:
					augmentedinputs3.append(flipimage(augimg))
					augmentedtargets3.append(target)
				rndnr+=1
		
		x = np.arange(-2,3,2)
		y = np.arange(-2,3,2)

		for stepx in x:
			for stepy in y:
				if stepy == 2:
				        augimg = img[:,:,2+stepy:]
				else:
				        augimg = img[:,:,2+stepy:-2+stepy]
				if stepx == 2:
				        augimg = augimg[:,2+stepx:,:]
				else:
				        augimg = augimg[:,2+stepx:-2+stepx,:]
				augimg = rescale28(augimg, 24,24)
				if rand[rndnr]==0:
					augmentedinputs0.append(augimg)
					augmentedtargets0.append(target)
				elif rand[rndnr]==1:
					augmentedinputs1.append(augimg)
					augmentedtargets1.append(target)
				elif rand[rndnr]==2:
					augmentedinputs2.append(augimg)
					augmentedtargets2.append(target)
				elif rand[rndnr]==3:
					augmentedinputs3.append(augimg)
					augmentedtargets3.append(target)
				rndnr+=1
				if rand[rndnr]==0:
					augmentedinputs0.append(flipimage(augimg))
					augmentedtargets0.append(target)
				elif rand[rndnr]==1:
					augmentedinputs1.append(flipimage(augimg))
					augmentedtargets1.append(target)
				elif rand[rndnr]==2:
					augmentedinputs2.append(flipimage(augimg))
					augmentedtargets2.append(target)
				elif rand[rndnr]==3:
					augmentedinputs3.append(flipimage(augimg))
					augmentedtargets3.append(target)
				rndnr+=1
		rndnr=0
		imgnr += 1
	print("saving")
	np.savez(data_dir + "aug0input" + str(batchnr) + '.npz', augmentedinputs0)
	np.savez(data_dir + "aug0target" + str(batchnr) + '.npz', augmentedtargets0)
	np.savez(data_dir + "aug1input" + str(batchnr) + '.npz', augmentedinputs1)
	np.savez(data_dir + "aug1target" + str(batchnr) + '.npz', augmentedtargets1)
	np.savez(data_dir + "aug2input" + str(batchnr) + '.npz', augmentedinputs2)
	np.savez(data_dir + "aug2target" + str(batchnr) + '.npz', augmentedtargets2)
	np.savez(data_dir + "aug3input" + str(batchnr) + '.npz', augmentedinputs3)
	np.savez(data_dir + "aug3target" + str(batchnr) + '.npz', augmentedtargets3)

def resizeimages(imgs):
	resimg = []
	for img in imgs:
		resimg.append(rescale(img,24,24))
	return np.asarray(resimg)



X_trainnoaug, y_trainnoaug, X_test, y_test = load3.cifar10pre(dtype=theano.config.floatX, grayscale=False)
print(X_trainnoaug.shape)

X_trainnoaug0, X_trainnoaug1, X_trainnoaug2, X_trainnoaug3, X_trainnoaug4 = X_trainnoaug[:10000], X_trainnoaug[10000:20000], X_trainnoaug[20000:30000], X_trainnoaug[30000:40000], X_trainnoaug[40000:50000]
y_trainnoaug0, y_trainnoaug1, y_trainnoaug2, y_trainnoaug3, y_trainnoaug4 = y_trainnoaug[:10000], y_trainnoaug[10000:20000], y_trainnoaug[20000:30000], y_trainnoaug[30000:40000], y_trainnoaug[40000:50000]

print(y_trainnoaug4.shape)

# Reshape data
X_trainnoaug0 = X_trainnoaug0.reshape((-1, 3, 32, 32))
X_trainnoaug1 = X_trainnoaug1.reshape((-1, 3, 32, 32))
X_trainnoaug2 = X_trainnoaug2.reshape((-1, 3, 32, 32))
X_trainnoaug3 = X_trainnoaug3.reshape((-1, 3, 32, 32))
X_trainnoaug4 = X_trainnoaug4.reshape((-1, 3, 32, 32))

print("start augmenting 0")
augmentdata(X_trainnoaug0, y_trainnoaug0, 0)
print("start augmenting 1")
augmentdata(X_trainnoaug1, y_trainnoaug1, 1)
print("start augmenting 2")
augmentdata(X_trainnoaug2, y_trainnoaug2, 2)
print("start augmenting 3")
augmentdata(X_trainnoaug3, y_trainnoaug3, 3)
print("start augmenting 4")
augmentdata(X_trainnoaug4, y_trainnoaug4, 4)
print("done")








