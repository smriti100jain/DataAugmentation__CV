from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import print_function
import keras
from keras import backend
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans.utils_mnist import data_mnist
from cleverhans.attacks import jsma
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes, cnn_model, pair_visual, grid_visual
from Utils.utils_data import data_pasta, data_mnist, data_cars
from Utils.utils_model import cnn_model, cnn_model_cars
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model, pair_visual
import os
from keras.utils import np_utils

import numpy as np
import pylab as plt
from matplotlib import pyplot
import time
import matplotlib.pyplot as plt
import pickle
import time
import scipy
from scipy import misc
np.random.seed(10)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


FLAGS = flags.FLAGS
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
blackwhite = int(raw_input('1. blackwhite 0: color'))
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_boolean('enable_vis', False, 'Enable sample visualization.')

if(blackwhite==1):
    flags.DEFINE_integer('nb_channels', 1, 'Nb of color channels in the input.')
else:
    flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')

datacode = int(raw_input("Enter a dataset code 1:mnist,2:pasta,3:car "))
def data_pasta(blackWhite,DataSetPath):
    # These values are specific to MNIST
    #DataSetPath = os.path.join('DataSets','Barilla_Database')
    #classes = os.listdir(DataSetPath)
    classes = [u'spaghettoni', u'napo-cerise', u'coquillette', u'farfalle', u'ricotta', u'pesti', u'napoletana', u'provencale', u'pipe-rigate', u'gnocchi', u'penne', u'pennette', u'lasagne']

    nb_classes = len(classes)
    if(blackWhite==1):
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


def data_pasta_GAN(blackWhite,DataSetPath,DataSetPath_GAN):
    # These values are specific to MNIST
    #DataSetPath = os.path.join('DataSets','Barilla_Database')
    #classes = os.listdir(DataSetPath)
    classes = [u'spaghettoni', u'napo-cerise', u'coquillette', u'farfalle', u'ricotta', u'pesti', u'napoletana', u'provencale', u'pipe-rigate', u'gnocchi', u'penne', u'pennette', u'lasagne']

    nb_classes = len(classes)
    if(blackWhite==1):
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
    X_adv_test = []
    y_adv_test = []
    for classLabel in range(len(classes)):
        print(classes[classLabel])
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        train_GAN_path = os.path.join(DataSetPath_GAN,'Train','1',classes[classLabel],'1')
        test_GAN_path = os.path.join(DataSetPath_GAN,'Test','1',classes[classLabel],'1')
        classSamples_train_GAN = os.listdir(train_GAN_path)
        classSamples_test_GAN = os.listdir(test_GAN_path)
        indices_GAN_train = np.random.permutation(len(classSamples_train_GAN))
        indices_GAN_test = np.random.permutation(len(classSamples_test_GAN))
     

        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
  

        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
            trainDataGAN = [scipy.misc.imresize(scipy.ndimage.imread(os.path.join(train_GAN_path,classSamples_train_GAN[i]),flatten=False),size=(img_rows,img_cols)) for i in indices_GAN_train]
            print(np.shape(trainData))
            testDataGAN = [scipy.misc.imresize(scipy.ndimage.imread(os.path.join(test_GAN_path,classSamples_test_GAN[i]),flatten=False),size=(img_rows,img_cols)) for i in indices_GAN_test]
            print(np.shape(trainDataGAN))
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==0):
            if(np.shape(trainDataGAN)[0]>0):
                trainData = np.vstack((trainData,trainDataGAN))
            X_train = trainData
            X_test = testData
            X_valid = validData
            X_adv_test = testDataGAN

        else:
            
            X_train = np.vstack((X_train,trainData))
            if(np.shape(trainDataGAN)[0]>0):
                X_train = np.vstack((X_train,trainDataGAN))
            X_test = np.vstack((X_test,testData))
            if(np.shape(testDataGAN)[0]>0):
                X_adv_test = np.vstack((X_adv_test,testDataGAN))
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,classLabel*np.ones(len(training_idx))))
        y_train = np.hstack((y_train,classLabel*np.ones(len(indices_GAN_train))))
        y_test = np.hstack((y_test,classLabel*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,classLabel*np.ones(len(valid_idx))))
        y_adv_test = np.hstack((y_adv_test,classLabel*np.ones(len(indices_GAN_test))))
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        print(X_adv_test)
        X_adv_test = X_adv_test.reshape(X_adv_test.shape[0],img_rows,img_cols,1)
        

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_adv_test = X_adv_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
    X_adv_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_valid.shape[0], 'valid samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    y_adv_test = np_utils.to_categorical(y_adv_test,nb_classes)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid, X_adv_test,y_adv_test

def data_cars(blackWhite,DataSetPath):
    # These values are specific to MNIST
    #DataSetPath = os.path.join('DataSets','Renault_Database')
    #classes = os.listdir(DataSetPath)
    classes = [u'bouchon_liquide_refroidissement', u'bouchon_liquide_freins', u'voyant_service', u'coffre_ferme', u'trappe_carburant_ouverte', u'volant_vitesse_droite', u'prise_aux', u'feux_arriere', u'carte_acces', u'clim_auto_reglage_vitesse', u'commodo_vitre', u'btn_aide_parking', u'poignee_porte', u'prise_12v', u'combine', u'calandre', u'fusibles', u'commande_son', u'roue', u'clim_auto_recyclage_air', u'antibrouillard_avant', u'securite_enfants', u'commodo_eclairage', u'bouchon_huile_moteur', u'reglage_feux_avant', u'feux_avant', u'btn_start_stop', u'clim_auto_on_off', u'lecteur_carte', u'btn_eco', u'bouchon_lave_vitres', u'clim_auto_degivrage', u'volant_vitesse_gauche', u'deverouillage_capot', u'clim_auto_voir_clair', u'clim_auto_reglage_repartition', u'batterie', u'antibrouillard_arriere', u'levier_vitesse', u'roue_secours', u'commande_eclairage', u'btn_deverouillage', u'essuie_glaces', u'btn_warning', u'btn_start', u'clim_auto_reglage_temp', u'panneau_retro', u'btn_regul_vitesse', u'panneau_vitres', u'ecran_tactile', u'trappe_carburant_fermee', u'coffre_ouvert']
    print(classes)
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
        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
        print(len(training_idx))
        print(len(test_idx))
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




def data_cars_GAN(blackWhite,DataSetPath,DataSetPath_GAN):
    # These values are specific to MNIST
    #DataSetPath = os.path.join('DataSets','Barilla_Database')
    #classes = os.listdir(DataSetPath)
    classes = [u'bouchon_liquide_refroidissement', u'bouchon_liquide_freins', u'voyant_service', u'coffre_ferme', u'trappe_carburant_ouverte', u'volant_vitesse_droite', u'prise_aux', u'feux_arriere', u'carte_acces', u'clim_auto_reglage_vitesse', u'commodo_vitre', u'btn_aide_parking', u'poignee_porte', u'prise_12v', u'combine', u'calandre', u'fusibles', u'commande_son', u'roue', u'clim_auto_recyclage_air', u'antibrouillard_avant', u'securite_enfants', u'commodo_eclairage', u'bouchon_huile_moteur', u'reglage_feux_avant', u'feux_avant', u'btn_start_stop', u'clim_auto_on_off', u'lecteur_carte', u'btn_eco', u'bouchon_lave_vitres', u'clim_auto_degivrage', u'volant_vitesse_gauche', u'deverouillage_capot', u'clim_auto_voir_clair', u'clim_auto_reglage_repartition', u'batterie', u'antibrouillard_arriere', u'levier_vitesse', u'roue_secours', u'commande_eclairage', u'btn_deverouillage', u'essuie_glaces', u'btn_warning', u'btn_start', u'clim_auto_reglage_temp', u'panneau_retro', u'btn_regul_vitesse', u'panneau_vitres', u'ecran_tactile', u'trappe_carburant_fermee', u'coffre_ouvert']

    nb_classes = len(classes)
    if(blackWhite==1):
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
    X_adv_test = []
    y_adv_test = []
    tot = 0
    for classLabel in range(len(classes)):
        print(classes[classLabel])
        classPath = os.path.join(DataSetPath,classes[classLabel])
        classSamples = os.listdir(classPath)
        numSamples = len(classSamples)
        indices = np.random.permutation(numSamples)
        train_GAN_path = os.path.join(DataSetPath_GAN,'Train','1',classes[classLabel],'0')
        test_GAN_path = os.path.join(DataSetPath_GAN,'Test','1',classes[classLabel],'1')
        classSamples_train_GAN = os.listdir(train_GAN_path)
        print('here'+str(len(classSamples_train_GAN)))
        tot = tot+len(classSamples_train_GAN)
        classSamples_test_GAN = os.listdir(test_GAN_path)
        indices_GAN_train = np.random.permutation(len(classSamples_train_GAN))
        indices_GAN_test = np.random.permutation(len(classSamples_test_GAN))
     

        training_idx, test_idx, valid_idx = indices[:int(0.6*numSamples)], indices[int(0.6*numSamples)+1:int(0.8*numSamples)], indices[int(0.8*numSamples)+1:]
  

        if(blackWhite==1):
            trainData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(rgb2gray(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False)),size=(img_rows,img_cols)) for i in valid_idx]
            trainDataGAN = [scipy.misc.imresize(scipy.ndimage.imread(os.path.join(train_GAN_path,classSamples_train_GAN[i]),flatten=False),size=(img_rows,img_cols)) for i in indices_GAN_train]
            print(np.shape(trainData))
            testDataGAN = [scipy.misc.imresize(scipy.ndimage.imread(os.path.join(test_GAN_path,classSamples_test_GAN[i]),flatten=False),size=(img_rows,img_cols)) for i in indices_GAN_test]
            print(np.shape(trainDataGAN))
        else:
            trainData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in training_idx]
            testData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in test_idx]
            validData =[scipy.misc.imresize(scipy.ndimage.imread(classPath+'/'+classSamples[i],flatten=False),size=(img_rows,img_cols)) for i in valid_idx]

        if(classLabel==0):
            if(np.shape(trainDataGAN)[0]>0):
                trainData = np.vstack((trainData,trainDataGAN))
            X_train = trainData
            X_test = testData
            X_valid = validData
            if(np.shape(testDataGAN)[0]>0):
                X_adv_test = testDataGAN

        else:
            
            X_train = np.vstack((X_train,trainData))
            if(np.shape(trainDataGAN)[0]>0):
                X_train = np.vstack((X_train,trainDataGAN))
            X_test = np.vstack((X_test,testData))

            if(np.shape(testDataGAN)[0]>0):
                if(np.shape(X_adv_test)[0]>0):
                    print('here-----------------------------------')
                    print(np.shape(testDataGAN))
                    X_adv_test = np.vstack((X_adv_test,testDataGAN))
                else:
                    X_adv_test = testDataGAN
            X_valid = np.vstack((X_valid,validData))
        y_train = np.hstack((y_train,classLabel*np.ones(len(training_idx))))
        y_train = np.hstack((y_train,classLabel*np.ones(len(indices_GAN_train))))
        y_test = np.hstack((y_test,classLabel*np.ones(len(test_idx))))
        y_valid = np.hstack((y_valid,classLabel*np.ones(len(valid_idx))))
        y_adv_test = np.hstack((y_adv_test,classLabel*np.ones(len(indices_GAN_test))))
    print('total='+str(tot))
    if(blackWhite==1):
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        X_adv_test = X_adv_test.reshape(X_adv_test.shape[0],img_rows,img_cols,1)
        

    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    X_adv_test = X_adv_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_valid /= 255
    X_adv_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_valid.shape[0], 'valid samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    y_adv_test = np_utils.to_categorical(y_adv_test,nb_classes)

    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid, X_adv_test,y_adv_test




if(datacode ==3 ):
    flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
    if(blackwhite==1):
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'cars.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs', 2, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 5, 'Size of training batches')
        flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
        flags.DEFINE_integer('img_rows', 100, 'Input row dimension')
        flags.DEFINE_integer('img_cols', 100, 'Input column dimension')
    else:
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'cars.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs', 340, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 5, 'Size of training batches')
        flags.DEFINE_float('learning_rate', 0.005, 'Learning rate for training')
        flags.DEFINE_integer('img_rows', 100, 'Input row dimension')
        flags.DEFINE_integer('img_cols', 100, 'Input column dimension')
elif(datacode ==2):
    flags.DEFINE_integer('nb_filters', 128, 'Number of convolutional filter to use')
    if(blackwhite==1):
        flags.DEFINE_integer('img_rows', 100, 'Input row dimension')
        flags.DEFINE_integer('img_cols', 100, 'Input column dimension')
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs',200, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 5, 'Size of training batches')
        flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
    else:
        flags.DEFINE_integer('img_rows', 100, 'Input row dimension')
        flags.DEFINE_integer('img_cols', 100, 'Input column dimension')
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 5, 'Size of training batches')
        flags.DEFINE_float('learning_rate', 0.005, 'Learning rate for training')

elif(datacode ==1):
    flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
    flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
    flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
    flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
    flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
    flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')

tf.set_random_seed(1234)
np.random.seed(10)
if(datacode == 1):
        #Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist()
        if(blackwhite==0):
            dataFolder = '../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExPapernot/mnist/'
        else:
            dataFolder = '../../opt/ADL_db/Users/sjain/adversarial-train/adversarialExPapernot/mnistB/'
        flags.DEFINE_integer('nb_classes', np.shape(Y_test)[1], 'Nb of classes.')
        


        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 30, FLAGS.nb_classes)
        label_smooth = .1
        Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
elif(datacode == 2):
        #Get pasta data
        DataSetPath = os.path.join('DataSets','Barilla_Database')
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = data_pasta(blackwhite,DataSetPath)
        DataSetPath_GAN = os.path.join('..','..','opt','ADL_db','Users','sjain','adversarial-train','adversarialExPapernot_vs2','PastaB')
        X_train_GAN, Y_train_GAN, X_test, Y_test, X_valid, Y_valid,X_test_adverse,Y_test_adverse  = data_pasta_GAN(blackwhite,DataSetPath,DataSetPath_GAN)
        print(np.shape(X_train_GAN))
        print(np.shape(Y_train_GAN))
        print('Data fetched --------------------------------------------------------------------------------------------------------------')
        if(blackwhite==0):
            dataFolder = 'adversarial-train/adversarialExPapernot/Pasta/'
        else:
            dataFolder = 'adversarial-train/adversarialExPapernot/PastaB/'
        flags.DEFINE_integer('nb_classes', np.shape(Y_test)[1], 'Nb of classes.')
        flags.DEFINE_integer('source_samples', np.shape(Y_test)[0], 'Nb of test set examples to attack')

        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 40, FLAGS.nb_classes)

elif(datacode == 3):
        DataSetPath = os.path.join('DataSets','Renault_Database')
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = data_cars(blackwhite,DataSetPath)
        DataSetPath_GAN = os.path.join('..','..','opt','ADL_db','Users','sjain','adversarial-train','adversarialExPapernot_vs2','CarsB')
        X_train_GAN, Y_train_GAN, X_test, Y_test, X_valid, Y_valid,X_test_adverse,Y_test_adverse = data_cars_GAN(blackwhite,DataSetPath,DataSetPath_GAN)
        
        print(np.shape(Y_test)[1])
        if(blackwhite==0):
            dataFolder = 'adversarial-train/adversarialExPapernot/Cars/'
        else:
            dataFolder = 'adversarial-train/adversarialExPapernot/CarsB/'
        flags.DEFINE_integer('nb_classes', np.shape(Y_test)[1], 'Nb of classes.')
        flags.DEFINE_integer('source_samples', np.shape(Y_test)[0], 'Nb of test set examples to attack')

        tf.set_random_seed(5)
        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 50, FLAGS.nb_classes)

flags.DEFINE_integer('source_samplesTest', np.shape(Y_test)[0], 'Nb of test set examples to attack')
flags.DEFINE_integer('source_samplesTrain', np.shape(Y_train)[0], 'Nb of test set examples to attack')  



def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """


    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)


    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)


    # Define input TF placeholder
    
    x = tf.placeholder(tf.float32, shape=(None, FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))
    
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

    # Define TF model graph
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model if it does not exist in the train_dir folder
    
    train_params = {
            'nb_epochs': FLAGS.nb_epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
    }
    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))
 
    model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate,args=train_params)
    print(np.shape(X_test))
    print(np.shape(Y_test))
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy1 = model_eval(sess, x, y, predictions, X_test, Y_test,args=eval_params)
    
    print(accuracy1)
    accuracy2 = model_eval(sess, x, y, predictions, X_test_adverse, Y_test_adverse,args=eval_params)
    

    print(accuracy2)

    # Define TF model graph
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model if it does not exist in the train_dir folder
    
    train_params = {
            'nb_epochs': FLAGS.nb_epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
    }
    print(np.shape(X_train_GAN))
    print(np.shape(Y_train_GAN))
    model_train(sess, x, y, predictions, X_train_GAN, Y_train_GAN, evaluate=evaluate, args=train_params)
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': FLAGS.batch_size}

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy3 = model_eval(sess, x, y, predictions, X_test, Y_test,args=eval_params)
    
    print('Test accuracy(Original) on legitimate test examples: {0}'.format(accuracy1))
    print('Test accuracy(Original) on adversarial test examples: {0}'.format(accuracy2))
    print('Test accuracy(After) on legitimate test examples: {0}'.format(accuracy3))

    accuracy4 = model_eval(sess, x, y, predictions, X_test_adverse, Y_test_adverse,args=eval_params)
    print('Test accuracy(After) on adversarial test examples: {0}'.format(accuracy4))

    sess.close()

    

if __name__ == '__main__':
    app.run()
