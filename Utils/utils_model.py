from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxoutDense, MaxPooling2D
from keras.layers import Convolution2D
from keras.layers.convolutional import ZeroPadding2D
from keras.constraints import maxnorm
import tensorflow as tf
import numpy as np
np.random.seed(1234)
def cnn_model(logits, input_ph, img_rows, img_cols, channels, nb_filters, nb_classes):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """

    model = Sequential()

    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)
    '''
    layers = [Dropout(0.2, input_shape=input_shape),
              Convolution2D(nb_filters, 8, 8,
                            subsample=(2, 2),
                            border_mode="same"
                            ),
              Activation('relu'),
              Convolution2D(nb_filters * 2, 6, 6, subsample=(2, 2),
                            border_mode="valid"),
              Activation('relu'),
              Convolution2D(nb_filters * 2, 5, 5, subsample=(1, 1)),
              Activation('relu'),
              Dropout(0.5),
              Flatten(),
              Dense(nb_classes)]
    '''
    layers = [Convolution2D(nb_filters, 8, 8,
                            subsample=(2, 2),
                            border_mode="same", input_shape=input_shape
                            ),
              Activation('relu'),
              Convolution2D(nb_filters * 2, 6, 6, subsample=(2, 2),
                            border_mode="valid"),
              Activation('relu'),
              Convolution2D(nb_filters * 2, 5, 5, subsample=(1, 1)),
              Activation('relu'),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        print(layer)
        model.add(layer)
    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model


def cnn_model_cars(logits, input_ph, img_rows, img_cols, channels, nb_filters, nb_classes):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
   


    model = Sequential()
    model.add(Convolution2D(nb_filters, 3, 3, input_shape=(img_rows, img_cols,channels), activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))
    if logits:
        logits_tensor = model(input_ph)
    if logits:
        return model, logits_tensor
    else:
        return model


#http://cbonnett.github.io/Insight.html
#using vgg network

def get_base_model(logits, input_ph, img_rows, img_cols, channels, nb_classes):
    """
    Returns the convolutional part of VGG net as a keras model 
    All layers have trainable set to False
    """
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape, name='image_input'))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    
    if logits:
        logits_tensor = model(input_ph)
    #model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model



from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
def trainedvgg(logits, input_ph, img_rows, img_cols, channels, nb_classes):
    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    model_vgg16_conv.summary()

    #Create your own input format (here 3x200x200)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)
    print(input_shape)
    input = Input(shape=input_shape,name = 'image_input')

    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(input)
    print(output_vgg16_conv)
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    #Create your own model 
    my_model = Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()

    return my_model