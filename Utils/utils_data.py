from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras.datasets import mnist
from keras.utils import np_utils
import warnings
import numpy as np
import scipy
from scipy import misc, ndimage
import os
import pickle

np.random.seed(10)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def data_mnist():
    """
    Preprocess MNIST dataset
    :return:
    """

    # These values are specific to MNIST
    img_rows = 28
    img_cols = 28
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test

def data_pasta(blackWhite,GAN):
    # These values are specific to MNIST
    DataSetPath = os.path.join('..','DataSets','Barilla_Database')
    classes = os.listdir(DataSetPath)
    nb_classes = len(classes)
    if(GAN==1):
        img_rows = 100
        img_cols = 100
    else:
        img_cols = 100
        img_rows = 100
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

        trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
        testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
        validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

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

    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

        

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_valid.shape[0], 'valid samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

def data_cars(blackWhite,GAN):
    # These values are specific to MNIST
    DataSetPath = os.path.join('..','DataSets','Renault_Database')
    classes = os.listdir(DataSetPath)
    nb_classes = len(classes)
    if(GAN==1):
        img_rows = 100
        img_cols = 100
    else:
        img_cols = 100
        img_rows = 100
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
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

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


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid





def data_cars_augmented(blackWhite):
    # These values are specific to MNIST
    if(blackWhite == 1):
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedData'
    else:    
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedDatacolor'
    if(os.path.exists(savePath)):
        X_train = pickle.load(open( os.path.join(savePath,'X_train.p'), "rb" ) )
        X_test = pickle.load(open( os.path.join(savePath,'X_test.p'), "rb" ) )
        X_valid = pickle.load(open( os.path.join(savePath,'X_valid.p'), "rb" ) )
        Y_train = pickle.load(open( os.path.join(savePath,'Y_train.p'), "rb" ) )
        Y_test = pickle.load(open( os.path.join(savePath,'Y_test.p'), "rb" ) )
        Y_valid = pickle.load(open( os.path.join(savePath,'Y_valid.p'), "rb" ) )
        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

    os.makedirs(savePath)

    DataSetPath = os.path.join('..','..','..','opt','exchange','nmonet','Ioan','Renault_Database_Augmented')
    classes = os.listdir(DataSetPath)
    nb_classes = len(classes)
    img_rows = 100
    img_cols = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(len(classes)):
        print(classLabel)
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.4*numSamples)], indices[int(0.4*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

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


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    pickle.dump(X_train, open( os.path.join(savePath,'X_train.p'), "wb" ) )
    pickle.dump(X_test, open( os.path.join(savePath,'X_test.p'), "wb" ) )
    pickle.dump(X_valid, open( os.path.join(savePath,'X_valid.p'), "wb" ) )
    pickle.dump(Y_train, open( os.path.join(savePath,'Y_train.p'), "wb" ) )
    pickle.dump(Y_test, open( os.path.join(savePath,'Y_test.p'), "wb" ) )
    pickle.dump(Y_valid, open( os.path.join(savePath,'Y_valid.p'), "wb" ) )
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


def data_cars_augmentedbeg13(blackWhite):
    # These values are specific to MNIST
    if(blackWhite == 1):
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedData13'
    else:    
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedDatacolor13'
    if(os.path.exists(savePath)):
        X_train = pickle.load(open( os.path.join(savePath,'X_train.p'), "rb" ) )
        X_test = pickle.load(open( os.path.join(savePath,'X_test.p'), "rb" ) )
        X_valid = pickle.load(open( os.path.join(savePath,'X_valid.p'), "rb" ) )
        Y_train = pickle.load(open( os.path.join(savePath,'Y_train.p'), "rb" ) )
        Y_test = pickle.load(open( os.path.join(savePath,'Y_test.p'), "rb" ) )
        Y_valid = pickle.load(open( os.path.join(savePath,'Y_valid.p'), "rb" ) )
        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

    os.makedirs(savePath)
    DataSetPath = os.path.join('..','..','..','opt','exchange','nmonet','Ioan','Renault_Database_Augmented')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    img_rows = 100
    img_cols = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(nb_classes):#range(len(classes)):
        print(classLabel)
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

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


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    pickle.dump(X_train, open( os.path.join(savePath,'X_train.p'), "wb" ) )
    pickle.dump(X_test, open( os.path.join(savePath,'X_test.p'), "wb" ) )
    pickle.dump(X_valid, open( os.path.join(savePath,'X_valid.p'), "wb" ) )
    pickle.dump(Y_train, open( os.path.join(savePath,'Y_train.p'), "wb" ) )
    pickle.dump(Y_test, open( os.path.join(savePath,'Y_test.p'), "wb" ) )
    pickle.dump(Y_valid, open( os.path.join(savePath,'Y_valid.p'), "wb" ) )
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


def data_cars13(blackWhite,GAN):
    # These values are specific to MNIST
    DataSetPath = os.path.join('..','DataSets','Renault_Database')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    if(GAN==1):
        img_rows = 100
        img_cols = 100
    else:
        img_cols = 100
        img_rows = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(nb_classes):#len(classes)):
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

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


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid




def data_cars_augmented26(blackWhite):
    # These values are specific to MNIST
    if(blackWhite == 1):
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedData26'
    else:    
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedDatacolor26'
    if(os.path.exists(savePath)):
        X_train = pickle.load(open( os.path.join(savePath,'X_train.p'), "rb" ) )
        X_test = pickle.load(open( os.path.join(savePath,'X_test.p'), "rb" ) )
        X_valid = pickle.load(open( os.path.join(savePath,'X_valid.p'), "rb" ) )
        Y_train = pickle.load(open( os.path.join(savePath,'Y_train.p'), "rb" ) )
        Y_test = pickle.load(open( os.path.join(savePath,'Y_test.p'), "rb" ) )
        Y_valid = pickle.load(open( os.path.join(savePath,'Y_valid.p'), "rb" ) )
        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

    os.makedirs(savePath)
    DataSetPath = os.path.join('..','..','..','opt','exchange','nmonet','Ioan','Renault_Database_Augmented')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    img_rows = 100
    img_cols = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(13,nb_classes+13):#range(len(classes)):
        print(classLabel)
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==13):
            X_train = trainData
            X_test = testData
            X_valid = validData

        else:
            X_train = np.vstack((X_train,trainData))
            X_test = np.vstack((X_test,testData))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,(classLabel-13)*np.ones(len(training_idx))))
        y_test = np.hstack((y_test,(classLabel-13)*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,(classLabel-13)*np.ones(len(valid_idx))))


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    pickle.dump(X_train, open( os.path.join(savePath,'X_train.p'), "wb" ) )
    pickle.dump(X_test, open( os.path.join(savePath,'X_test.p'), "wb" ) )
    pickle.dump(X_valid, open( os.path.join(savePath,'X_valid.p'), "wb" ) )
    pickle.dump(Y_train, open( os.path.join(savePath,'Y_train.p'), "wb" ) )
    pickle.dump(Y_test, open( os.path.join(savePath,'Y_test.p'), "wb" ) )
    pickle.dump(Y_valid, open( os.path.join(savePath,'Y_valid.p'), "wb" ) )
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


def data_cars26(blackWhite,GAN):
    # These values are specific to MNIST
    DataSetPath = os.path.join('..','DataSets','Renault_Database')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    if(GAN==1):
        img_rows = 100
        img_cols = 100
    else:
        img_cols = 100
        img_rows = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(13,nb_classes+13):#len(classes)):
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==13):
            X_train = trainData
            X_test = testData
            X_valid = validData

        else:
            X_train = np.vstack((X_train,trainData))
            X_test = np.vstack((X_test,testData))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,(classLabel-13)*np.ones(len(training_idx))))
        y_test = np.hstack((y_test,(classLabel-13)*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,(classLabel-13)*np.ones(len(valid_idx))))


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


def data_cars_augmented39(blackWhite):
    # These values are specific to MNIST
    if(blackWhite == 1):
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedData39'
    else:    
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedDatacolor39'
    if(os.path.exists(savePath)):
        X_train = pickle.load(open( os.path.join(savePath,'X_train.p'), "rb" ) )
        X_test = pickle.load(open( os.path.join(savePath,'X_test.p'), "rb" ) )
        X_valid = pickle.load(open( os.path.join(savePath,'X_valid.p'), "rb" ) )
        Y_train = pickle.load(open( os.path.join(savePath,'Y_train.p'), "rb" ) )
        Y_test = pickle.load(open( os.path.join(savePath,'Y_test.p'), "rb" ) )
        Y_valid = pickle.load(open( os.path.join(savePath,'Y_valid.p'), "rb" ) )
        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

    os.makedirs(savePath)
    DataSetPath = os.path.join('..','..','..','opt','exchange','nmonet','Ioan','Renault_Database_Augmented')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    img_rows = 100
    img_cols = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(26,nb_classes+26):#range(len(classes)):
        print(classLabel)
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==26):
            X_train = trainData
            X_test = testData
            X_valid = validData

        else:
            X_train = np.vstack((X_train,trainData))
            X_test = np.vstack((X_test,testData))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,(classLabel-26)*np.ones(len(training_idx))))
        y_test = np.hstack((y_test,(classLabel-26)*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,(classLabel-26)*np.ones(len(valid_idx))))


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    pickle.dump(X_train, open( os.path.join(savePath,'X_train.p'), "wb" ) )
    pickle.dump(X_test, open( os.path.join(savePath,'X_test.p'), "wb" ) )
    pickle.dump(X_valid, open( os.path.join(savePath,'X_valid.p'), "wb" ) )
    pickle.dump(Y_train, open( os.path.join(savePath,'Y_train.p'), "wb" ) )
    pickle.dump(Y_test, open( os.path.join(savePath,'Y_test.p'), "wb" ) )
    pickle.dump(Y_valid, open( os.path.join(savePath,'Y_valid.p'), "wb" ) )
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


def data_cars39(blackWhite,GAN):
    # These values are specific to MNIST
    DataSetPath = os.path.join('..','DataSets','Renault_Database')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    if(GAN==1):
        img_rows = 100
        img_cols = 100
    else:
        img_cols = 100
        img_rows = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(26,nb_classes+26):#len(classes)):
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==26):
            X_train = trainData
            X_test = testData
            X_valid = validData

        else:
            X_train = np.vstack((X_train,trainData))
            X_test = np.vstack((X_test,testData))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,(classLabel-26)*np.ones(len(training_idx))))
        y_test = np.hstack((y_test,(classLabel-26)*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,(classLabel-26)*np.ones(len(valid_idx))))


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


def data_cars_augmented52(blackWhite):
    # These values are specific to MNIST
    if(blackWhite == 1):
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedData52'
    else:    
        savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedDatacolor52'
    if(os.path.exists(savePath)):
        X_train = pickle.load(open( os.path.join(savePath,'X_train.p'), "rb" ) )
        X_test = pickle.load(open( os.path.join(savePath,'X_test.p'), "rb" ) )
        X_valid = pickle.load(open( os.path.join(savePath,'X_valid.p'), "rb" ) )
        Y_train = pickle.load(open( os.path.join(savePath,'Y_train.p'), "rb" ) )
        Y_test = pickle.load(open( os.path.join(savePath,'Y_test.p'), "rb" ) )
        Y_valid = pickle.load(open( os.path.join(savePath,'Y_valid.p'), "rb" ) )
        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

    os.makedirs(savePath)
    DataSetPath = os.path.join('..','..','..','opt','exchange','nmonet','Ioan','Renault_Database_Augmented')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    img_rows = 100
    img_cols = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(39,nb_classes+39):#range(len(classes)):
        print(classLabel)
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==39):
            X_train = trainData
            X_test = testData
            X_valid = validData

        else:
            X_train = np.vstack((X_train,trainData))
            X_test = np.vstack((X_test,testData))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,(classLabel-39)*np.ones(len(training_idx))))
        y_test = np.hstack((y_test,(classLabel-39)*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,(classLabel-39)*np.ones(len(valid_idx))))


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    pickle.dump(X_train, open( os.path.join(savePath,'X_train.p'), "wb" ) )
    pickle.dump(X_test, open( os.path.join(savePath,'X_test.p'), "wb" ) )
    pickle.dump(X_valid, open( os.path.join(savePath,'X_valid.p'), "wb" ) )
    pickle.dump(Y_train, open( os.path.join(savePath,'Y_train.p'), "wb" ) )
    pickle.dump(Y_test, open( os.path.join(savePath,'Y_test.p'), "wb" ) )
    pickle.dump(Y_valid, open( os.path.join(savePath,'Y_valid.p'), "wb" ) )
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


def data_cars52(blackWhite,GAN):
    # These values are specific to MNIST
    DataSetPath = os.path.join('..','DataSets','Renault_Database')
    classes = os.listdir(DataSetPath)
    nb_classes = 13#len(classes)
    if(GAN==1):
        img_rows = 100
        img_cols = 100
    else:
        img_cols = 100
        img_rows = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(39,nb_classes+39):#len(classes)):
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==39):
            X_train = trainData
            X_test = testData
            X_valid = validData

        else:
            X_train = np.vstack((X_train,trainData))
            X_test = np.vstack((X_test,testData))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,(classLabel-39)*np.ones(len(training_idx))))
        y_test = np.hstack((y_test,(classLabel-39)*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,(classLabel-39)*np.ones(len(valid_idx))))


        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid





def testingData(blackWhite):
    # These values are specific to MNIST
    savePath = '../../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExGoodFellow/CarsAugmentedDataTesting'
    
    if(blackWhite == 0):
        if(os.path.exists(os.path.join(savePath,'Color'))):
            X_train = pickle.load(open( os.path.join(savePath,'Color','X_train_Test.p'), "rb" ) )
            Y_train = pickle.load(open( os.path.join(savePath,'Color','Y_train_Test.p'), "rb" ) )
            return X_train, Y_train
    else:
        if(os.path.exists(savePath)):
            X_train = pickle.load(open( os.path.join(savePath,'X_train_Test.p'), "rb" ) )
        
            Y_train = pickle.load(open( os.path.join(savePath,'Y_train_Test.p'), "rb" ) )
        
            return X_train, Y_train

    os.makedirs(os.path.join(savePath,'Color'))

    DataSetPath = os.path.join('..','..','..','opt','exchange','nmonet','Ioan','Renault_Database_Augmented')
    classes = os.listdir(DataSetPath)
    nb_classes = len(classes)
    img_rows = 100
    img_cols = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(len(classes)):
        print(classLabel)
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx = indices[:int(1*numSamples)]
        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            

        if(classLabel==0):
            X_train = trainData
            
        else:
            X_train = np.vstack((X_train,trainData))
            
        y_train = np.hstack((y_train,classLabel*np.ones(len(training_idx))))
        

        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        

    
    X_train = X_train.astype('float32')
    
    X_train /= 255
    
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    
    if(blackWhite == 1):
        pickle.dump(X_train, open( os.path.join(savePath,'X_train_Test.p'), "wb" ) )
    
        pickle.dump(Y_train, open( os.path.join(savePath,'Y_train_Test.p'), "wb" ) )
    else:
        pickle.dump(X_train, open( os.path.join(savePath,'Color','X_train_Test.p'), "wb" ) )
    
        pickle.dump(Y_train, open( os.path.join(savePath,'Color','Y_train_Test.p'), "wb" ) )


    return X_train, Y_train




def pastaGAN(blackWhite,modelno):
    # These values are specific to MNIST
    DataSetPath = '../../../opt/ADL_db/Users/sjain/adversarial-train/GAN/DCGAN/PastaB_noise6_'+str(modelno)
    DataSetPath1 = os.path.join('..','DataSets','Barilla_Database')
    classes = os.listdir(DataSetPath1)
    
    nb_classes = len(classes)
    img_rows = 100
    img_cols = 100
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
        training_idx = indices[:int(1*numSamples)]
       
        trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
    
        if(classLabel==0):
            X_train = trainData
            
        else:
            X_train = np.vstack((X_train,trainData))
            
        y_train = np.hstack((y_train,classLabel*np.ones(len(training_idx))))
        

        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        

    
    X_train = X_train.astype('float32')
    
    X_train /= 255
    
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    


    return X_train, Y_train



def carsGAN(blackWhite,modelno):
    # These values are specific to MNIST
    DataSetPath = '../../../opt/ADL_db/Users/sjain/adversarial-train/GAN/DCGAN/Cars_noise8'+str(modelno)
    
    DataSetPath1 = os.path.join('..','DataSets','Renault_Database')
    classes = os.listdir(DataSetPath1)
    nb_classes = len(classes)
    img_rows = 100
    img_cols = 100
    X_train = []
    X_test = []
    X_valid = []
    y_test = []
    y_train = []
    y_valid = []
    for classLabel in range(len(classes)):
        print(classLabel)
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        training_idx = indices[:int(1*numSamples)]
       
        trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
    
        if(classLabel==0):
            X_train = trainData
            
        else:
            X_train = np.vstack((X_train,trainData))
            
        y_train = np.hstack((y_train,classLabel*np.ones(len(training_idx))))
        

        
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        

    
    X_train = X_train.astype('float32')
    
    X_train /= 255
    
   

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    


    return X_train, Y_train



