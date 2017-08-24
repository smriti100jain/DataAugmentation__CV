#! /usr/bin/python -u
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import print_function
import keras
from keras import backend
import numpy as np
import os
import sys
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import jsma
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes, cnn_model, pair_visual, grid_visual
from Utils.utils_data import data_pasta, data_mnist, data_cars
from Utils.utils_model import cnn_model, cnn_model_cars
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model, pair_visual
import os
import numpy as np
import pylab as plt
from matplotlib import pyplot
import time
import matplotlib.pyplot as plt
import pickle
import time
import scipy
from scipy import misc
FLAGS = flags.FLAGS
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
blackwhite = int(raw_input('1. blackwhite 0: color'))
#blackwhite = 1
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_boolean('enable_vis', False, 'Enable sample visualization.')

if(blackwhite==1):
    flags.DEFINE_integer('nb_channels', 1, 'Nb of color channels in the input.')
else:
    flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')

#data code and train or test to generate papernot adversarial examples

datacode = int(raw_input("Enter a dataset code 1:mnist,2:pasta,3:car "))
Train = int(raw_input("Press 1 if Adversarial Examples corresponding to Train Examples else 0"))
if(datacode ==3 ):
    flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
    if(blackwhite==1):
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'cars.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs', 340, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 5, 'Size of training batches')
        flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
        flags.DEFINE_integer('img_rows',100, 'Input row dimension')
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
        flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
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
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
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
            dataFolder = '../sjain/adversarial-train/adversarialExPapernot_vs2/mnist/'
        else:
            dataFolder = '../sjain/adversarial-train/adversarialExPapernot_vs2/mnistB/'
        flags.DEFINE_integer('nb_classes', np.shape(Y_test)[1], 'Nb of classes.')
        classes = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'10']



        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 30, FLAGS.nb_classes)
        label_smooth = .1
        Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
elif(datacode == 2):
        #Get pasta data
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = data_pasta(blackwhite)
        print('Data fetched --------------------------------------------------------------------------------------------------------------')
        if(blackwhite==0):
            dataFolder = '../sjain/adversarial-train/adversarialExPapernot_vs2/Pasta/'
        else:
            dataFolder = '../sjain/adversarial-train/adversarialExPapernot_vs2/PastaB/'
        flags.DEFINE_integer('nb_classes', np.shape(Y_test)[1], 'Nb of classes.')

        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 40, FLAGS.nb_classes)
        np.random.seed(10)
        indices = np.random.permutation(np.shape(X_train)[0])
        classes = [u'spaghettoni', u'napo-cerise', u'coquillette', u'farfalle', u'ricotta', u'pesti', u'napoletana', u'provencale', u'pipe-rigate', u'gnocchi', u'penne', u'pennette', u'lasagne']

        X_train = X_train[indices]
        Y_train = Y_train[indices]
elif(datacode == 3):
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = data_cars(blackwhite)
        if(blackwhite==0):
            dataFolder = '../sjain/adversarial-train/adversarialExPapernot_vs2/Cars/'
        else:
            dataFolder = '../sjain/adversarial-train/adversarialExPapernot_vs2/CarsB/'
        flags.DEFINE_integer('nb_classes', np.shape(Y_test)[1], 'Nb of classes.')

        tf.set_random_seed(5)
        classes = [u'bouchon_liquide_refroidissement', u'bouchon_liquide_freins', u'voyant_service', u'coffre_ferme', u'trappe_carburant_ouverte', u'volant_vitesse_droite', u'prise_aux', u'feux_arriere', u'carte_acces', u'clim_auto_reglage_vitesse', u'commodo_vitre', u'btn_aide_parking', u'poignee_porte', u'prise_12v', u'combine', u'calandre', u'fusibles', u'commande_son', u'roue', u'clim_auto_recyclage_air', u'antibrouillard_avant', u'securite_enfants', u'commodo_eclairage', u'bouchon_huile_moteur', u'reglage_feux_avant', u'feux_avant', u'btn_start_stop', u'clim_auto_on_off', u'lecteur_carte', u'btn_eco', u'bouchon_lave_vitres', u'clim_auto_degivrage', u'volant_vitesse_gauche', u'deverouillage_capot', u'clim_auto_voir_clair', u'clim_auto_reglage_repartition', u'batterie', u'antibrouillard_arriere', u'levier_vitesse', u'roue_secours', u'commande_eclairage', u'btn_deverouillage', u'essuie_glaces', u'btn_warning', u'btn_start', u'clim_auto_reglage_temp', u'panneau_retro', u'btn_regul_vitesse', u'panneau_vitres', u'ecran_tactile', u'trappe_carburant_fermee', u'coffre_ouvert']

        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 50, FLAGS.nb_classes)
        np.random.seed(10)
        indices = np.random.permutation(np.shape(X_train)[0])
        X_train = X_train[indices]
        Y_train = Y_train[indices]
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
    sess = tf.Session(config=config)
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

    model_train(sess, x, y, predictions, X_train, Y_train, args=train_params)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    print('\n')

        ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(FLAGS.source_samplesTest) + ' * ' +
          str(FLAGS.nb_classes) + ' adversarial examples')
    print('\n')

    # This array indicates whether an adversarial example was found for each
    # test set sample and target class
    results = np.zeros((FLAGS.nb_classes, FLAGS.source_samplesTest), dtype='i')

    # This array contains the fraction of perturbed features for each test set
    # sample and target class
    perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samplesTest),
                             dtype='f')

    # Define the TF graph for the model's Jacobian
    grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

    
    theta = +1

    #Train = 0

    if(Train == 0):

    

        for target in range(0,FLAGS.nb_classes):
            createFold = os.path.join(dataFolder,'Test',str(theta),str(classes[target]),str(0))
            createFold1 = os.path.join(dataFolder,'Test',str(theta),str(classes[target]),str(1))
            if not os.path.exists(createFold1):
                    os.makedirs(createFold)
                    os.makedirs(createFold1)
            # Loop over the samples we want to perturb into adversarial examples

        
        for sample_ind in xrange(0, FLAGS.source_samplesTest):


            # We want to find an adversarial example for each possible target class
            # (i.e. all classes that differ from the label given in the dataset)
            current_class = int(np.argmax(Y_test[sample_ind]))
            
            target_classes = other_classes(FLAGS.nb_classes, current_class)

           
            start_time = time.time()
            # Loop over all target classes
            for target in target_classes:
                # This call runs the Jacobian-based saliency map approach
                adv_x, res, percent_perturb = jsma(sess, x, predictions, grads,
                                                   X_test[sample_ind:
                                                          (sample_ind+1)],
                                                   target, theta=theta, gamma=0.10,
                                                   increase=True, back='tf',
                                                   clip_min=0, clip_max=1)
                if(FLAGS.nb_channels==1):
                    scipy.misc.imsave(os.path.join(dataFolder,'Test',str(theta),str(classes[current_class]),str(res),str(classes[target])+'.png'), np.reshape(adv_x,(FLAGS.img_rows, FLAGS.img_cols)))
                else:
                    scipy.misc.imsave(os.path.join(dataFolder,'Test',str(theta),str(classes[current_class]),str(res),str(classes[target])+'.png'), np.reshape(adv_x,(FLAGS.img_rows, FLAGS.img_cols,3)))
               
                # Update the arrays for later analysis
                results[target, sample_ind] = res
                perturbations[target, sample_ind] = percent_perturb
        

        # Compute the number of adversarial examples that were successfuly found
        nb_targets_tried = ((FLAGS.nb_classes - 1) * FLAGS.source_samplesTest)
        succ_rate = float(np.sum(results)) / nb_targets_tried
        
        print('\n')
        print('Avg. rate of successful adv. examples {0:.2f}'.format(succ_rate))
        np.save(os.path.join(dataFolder,'Test','successful.npy'), succ_rate)

        # Compute the average distortion introduced by the algorithm
        percent_perturbed = np.mean(perturbations)
        print('\n')
        print('Avg. rate of perturbed features {0:.2f}'.format(percent_perturbed))
        np.save(os.path.join(dataFolder,'Test','percent_perturbed.npy'), percent_perturbed)

    
    
    if(Train == 1):

    
        print('\n')
        print('Crafting ' + str(FLAGS.source_samplesTrain) + ' * ' +
              str(FLAGS.nb_classes) + ' adversarial examples')

        # This array indicates whether an adversarial example was found for each
        # test set sample and target class
        results = np.zeros((FLAGS.nb_classes, FLAGS.source_samplesTrain), dtype='i')

        # This array contains the fraction of perturbed features for each test set
        # sample and target class
        perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samplesTrain),
                                 dtype='f')

        # Define the TF graph for the model's Jacobian
        grads = jacobian_graph(predictions, x, FLAGS.nb_classes)


        for target in range(0,FLAGS.nb_classes):
            createFold = os.path.join(dataFolder,'Train',str(theta),str(classes[target]),str(0))
            createFold1 = os.path.join(dataFolder,'Train',str(theta),str(classes[target]),str(1))
            if not os.path.exists(createFold1):
                    os.makedirs(createFold)
                    os.makedirs(createFold1)
        
        # Loop over the samples we want to perturb into adversarial examples
        for sample_ind in xrange(0, FLAGS.source_samplesTrain):
            # We want to find an adversarial example for each possible target class
            # (i.e. all classes that differ from the label given in the dataset)
            current_class = int(np.argmax(Y_train[sample_ind]))
            
            target_classes = other_classes(FLAGS.nb_classes, current_class)
            start_time = time.time()
            # Loop over all target classes
            for target in target_classes:
                # This call runs the Jacobian-based saliency map approach
                adv_x, res, percent_perturb = jsma(sess, x, predictions, grads,
                                                   X_train[sample_ind:
                                                          (sample_ind+1)],
                                                   target, theta=theta, gamma=0.10,
                                                   increase=True, back='tf',
                                                   clip_min=0, clip_max=1)
                # Display the original and adversarial images side-by-side
                
                                    
               
                if(FLAGS.nb_channels==1):
                    scipy.misc.imsave(os.path.join(dataFolder,'Train',str(theta),str(classes[current_class]),str(res),str(classes[target])+'.png'), np.reshape(adv_x,(FLAGS.img_rows, FLAGS.img_cols)))
                else:
                    scipy.misc.imsave(os.path.join(dataFolder,'Train',str(theta),str(classes[current_class]),str(res),str(classes[target])+'.png'), np.reshape(adv_x,(FLAGS.img_rows, FLAGS.img_cols,3)))
               
               
                # Update the arrays for later analysis
                results[target, sample_ind] = res
                perturbations[target, sample_ind] = percent_perturb
        print('\n')
        print("--- %s seconds ---" % (time.time() - start_time))
        # Compute the number of adversarial examples that were successfuly found
        nb_targets_tried = ((FLAGS.nb_classes - 1) * FLAGS.source_samplesTrain)
        succ_rate = float(np.sum(results)) / nb_targets_tried
        print('Avg. rate of successful adv. examples {0:.2f}'.format(succ_rate))
        print('\n')
        # Compute the average distortion introduced by the algorithm
        percent_perturbed = np.mean(perturbations)
        print('\n')
        print('Avg. rate of perturbed features {0:.2f}'.format(percent_perturbed))
        print('\n')
        np.save(os.path.join(dataFolder,'Train','successful.npy'), succ_rate)
        np.save(os.path.join(dataFolder,'Train','percent_perturbed.npy'), percent_perturbed)
        # Close TF session
    
    sess.close()

    

if __name__ == '__main__':
    app.run()
