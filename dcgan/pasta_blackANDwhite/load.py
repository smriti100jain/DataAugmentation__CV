import sys

import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt

from lib.data_utils import shuffle
from lib.config import data_dir
import keras
from keras.datasets import mnist
from keras.utils import np_utils
import warnings
import numpy as np
import scipy
from scipy import misc, ndimage
import os

np.random.seed(10)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def pastaBlackWhite():
    # These values are specific to MNIST
    DataSetPath = os.path.join('..','..','DataSets','Barilla_Database')
    classes = os.listdir(DataSetPath)
    print(classes)
    nb_classes = len(classes)
    img_rows = 200
    img_cols = 200
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(len(classes)):
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]

        trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
        testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
        validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        
        if(classLabel==0):
            X_train = trainData
            X_test = testData
            X_valid = validData

        else:
            X_train = np.vstack((X_train,trainData))
            X_test = np.vstack((X_test,testData))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,classLabel*np.ones(len(training_idx))))
        y_test = np.hstack((y_test,classLabel*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,classLabel*np.ones(len(valid_idx))))


        

    X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows*img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_valid.shape[0], 'valid samples')

    
    return X_train, X_valid, X_test, y_train.astype('int'), y_valid.astype('int'), y_test.astype('int')
