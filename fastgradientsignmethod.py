from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

# Utils.utils_data has functions for getting respective train, test and validation datasets
from Utils.utils_data import data_pasta, data_mnist, data_cars, data_cars_augmented, testingData

# Utils.utils_model have functions of different classification models tested on original training set
from Utils.utils_model import cnn_model, cnn_model_cars,get_base_model,trainedvgg


#from cleverhans.utils import cnn_model, pair_visual
#training and evaluation functions from cleverhans toolbox to train a tensorflow classifier 
from cleverhans.utils_tf import model_train, model_eval, batch_eval


#Function which creates adversarial examples by giving a cost function
from cleverhans.attacks import fgsm
import os
import numpy as np
import pylab as plt
from matplotlib import pyplot
import time
import matplotlib.pyplot as plt
import pickle
import scipy
from scipy import misc
start_time = time.time()
FLAGS = flags.FLAGS

#destination folder to save the results
destinationFolder = '../sjain/adversarial-train/adversarialExGoodFellow/'
if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)



blackwhite = int(raw_input('Enter 1 if blackwhite else 0 '))
datacode = int(raw_input("Enter a dataset code 1:mnist,2:pasta,3:car,4:caraugmented "))
Training = int(raw_input("Enter 1 if you want to train at each epsilon(Very time consuming) else just adversarial examples will be stored "))
Testing = int(raw_input("Enter 1 if you want to test your model on some other data and new adversarial samples"))

if(Testing == 1):
    dataTestingX, dataTestingY = testingData(blackwhite)
    saveFolderTest = os.path.join(destinationFolder,'CarsAugmentedAdvOrig')
    if(blackwhite==1):
        saveFolderTest = os.path.join(saveFolderTest,'bw')
    else:
        saveFolderTest = os.path.join(saveFolderTest,'color')

if(blackwhite==1):
    flags.DEFINE_integer('nb_channels', 1, 'Nb of color channels in the input.')
else:
    flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')


#Different parameters and image sizes for different datasets
if(datacode ==4 ):
    #cars augmented
    if(blackwhite==1):
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'cars.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 128*2, 'Size of training batches')
        flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')
        flags.DEFINE_integer('img_rows', 100, 'Input row dimension')
        flags.DEFINE_integer('img_cols', 100, 'Input column dimension')

    else:
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'cars.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs', 340, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 16, 'Size of training batches')
        flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
        flags.DEFINE_integer('img_rows', 100, 'Input row dimension')
        flags.DEFINE_integer('img_cols', 100, 'Input column dimension')
if(datacode ==3 ):
    #cars
    if(blackwhite==1):
        flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
        flags.DEFINE_string('filename', 'cars.ckpt', 'Filename to save model under.')
        flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
        flags.DEFINE_integer('batch_size', 10, 'Size of training batches')
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
    #pasta
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
    #mnist
    flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
    flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
    flags.DEFINE_integer('nb_epochs', 100, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
    flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
    flags.DEFINE_integer('img_cols', 28, 'Input column dimension')



def plot_figures(figures, nrows, ncols,savefolder, savename,channels,eps):
    """Plot a dictionary of adversarial images with different epsilon for one sample.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    savefolder : folder to save the image
    savename : name of the image to save
    channels : for color image=3 and for blackWhite = 1
    eps : list of epsilon values for which there are adversarial images in dictionary 'figures'
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    i = 1
    title1 = 'eps=0'
    for ind,title in zip(range(len(figures)), figures):
        
        print(title1)
        if(channels == 1):
            axeslist.ravel()[ind].imshow(figures[str(title1)], cmap=plt.gray())
        else:
            axeslist.ravel()[ind].imshow(figures[str(title1)])
        axeslist.ravel()[ind].set_title(title1)
        axeslist.ravel()[ind].set_axis_off()
        if(i == len(eps)):
            break
        title1 = 'eps='+str(eps[i])
        i = i+1
    plt.show()
    fig.savefig(os.path.join(savefolder,savename))
    plt.close(fig)


def main(argv=None):

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("Gives an error if keras is not configured to use tensorflow.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    #Fetch train and test data for different datasets
    if(datacode == 1):
        #Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist()

        #Folder to save adversarial images corresponding to MNIST dataset
        if(blackwhite==0):
            dataFolder = os.path.join(destinationFolder ,'mnist')
        else:
            dataFolder = os.path.join(destinationFolder ,'mnistB')

        nb_classes = np.shape(Y_test)[1]
        #classification model: can modify it in Utils.utils_model
        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 30, nb_classes)
        #label smoothening to improve the stability of training
        label_smooth = .1
        Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
        classes = range(10)

    elif(datacode == 2):
        #Get pasta data set
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = data_pasta(blackwhite,0)

        #Folder to save adversarial images
        if(blackwhite==0):
            dataFolder = os.path.join(destinationFolder ,'Pasta')
        else:
            dataFolder = os.path.join(destinationFolder ,'PastaB')
        #Number of classes in case of pasta, it is 13
        nb_classes = np.shape(Y_test)[1]

        #Classification model
        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 40, nb_classes)
        DataSetPath = os.path.join('..','DataSets','Barilla_Database')
        classes = os.listdir(DataSetPath)

    elif(datacode == 3):
        #Get Cars data set
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = data_cars(blackwhite,0)

        #Folder to save adversarial images
        if(blackwhite==0):
            dataFolder = os.path.join(destinationFolder ,'Cars')
        else:
            dataFolder = os.path.join(destinationFolder ,'CarsB')
        nb_classes = np.shape(Y_test)[1]
        tf.set_random_seed(5)

        # Classification model
        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 80, nb_classes)
        
        #Path for dataset
        DataSetPath = os.path.join('..','DataSets','Renault_Database')
        classes = os.listdir(DataSetPath)

    elif(datacode == 4):
        #Get Cars augmented dataset using traditional data augmentation
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = data_cars_augmented(blackwhite)

        if(blackwhite==0):
            dataFolder = os.path.join(destinationFolder ,'CarsAugmented')
        else:
            dataFolder = os.path.join(destinationFolder ,'CarsAugmentedB')
        nb_classes = np.shape(Y_test)[1]
        
        tf.set_random_seed(5)
        model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 50, nb_classes)
        #model = get_base_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, nb_classes)
        #model = trainedvgg(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, nb_classes)
        
        #Path for dataset
        DataSetPath = os.path.join('..','..','..','opt','exchange','nmonet','Ioan','Renault_Database_Augmented')
        classes = os.listdir(DataSetPath)



    #Create Path to store adversarial images corresponding to train and test datasets

    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    
    if not os.path.exists(dataFolder):
        os.makedirs(os.path.join(dataFolder,'Train'))
        os.makedirs(os.path.join(dataFolder,'Test'))

    

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))
    
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    predictions = model(x)
    print("Defined TensorFlow model graph.")


    def evaluate():
        # Evaluate the accuracy of the Classification model on legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))
       

    # Train an MNIST model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }

   
    eval_params = {'batch_size': FLAGS.batch_size}



    #list of epsilon on which model is tested -------can reduce if reduce the computational time.
    eps = np.arange(0,0.5,0.01)
     
    #Classification model training on legitimate training samples
    model_train(sess, x, y, predictions, X_train, Y_train,evaluate=evaluate, args=train_params)

     
    accOrigAd = [None]*(len(eps))
    accAdWhole = [None]*(len(eps))
    accAdTest = [None]*(len(eps))
    accAdAd = [None]*(len(eps))
    

    def visualize(advers, original, label,pathName):
        #visualizing adversarial samples correspoinding to original samples with perturbations
        uniques = np.array(range(0,nb_classes))
        clasLab = uniques[label.argmax(1)]
        for cl in uniques:
            indd = list(np.where(clasLab==cl)[0])[0:10]
            savename = os.path.join(pathName,str(cl))
            if not os.path.exists(savename):
                os.makedirs(savename)
            for sample_ind in indd:#np.shape(advers)[0]):
                if(FLAGS.nb_channels==1):
                    adversarial = np.reshape(advers[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols))
                    orig = np.reshape(original[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols))
                else:
                    adversarial = np.reshape(advers[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))
                    orig = np.reshape(original[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))

                plt.ioff()
                figure = plt.figure()
                figure.canvas.set_window_title('Cleverhans: Pair Visualization')

                # Add the images to the plot
                perterbations = adversarial - orig
                for index, image in enumerate((orig, perterbations1, adversarial)):
                    figure.add_subplot(1, 3, index + 1)
                    
                    # If the image is 2D, then we have 1 color channel
                    if len(image.shape) == 2:
                        plt.imshow(image, cmap='gray')
                    else:
                        plt.imshow(image)
                figure.savefig(os.path.join(savename,str(sample_ind)+'.png'))
                plt.close(figure)

    

    # Evaluate the accuracy of the MNIST model on Test Samples
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)



    if(Testing!=1):
        accOrigAd[0] = accuracy
        accAdTest[0] = accuracy
        accAdAd[0] = accuracy
        
        #To visualize the variations of sample 1 with epsilon
        sample_ind = 1
        number_of_im = len(eps)
        figures = {}
        if(FLAGS.nb_channels==1):
            figures['im_0']=np.reshape(X_train[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols))
        else:
            figures['im_0']=np.reshape(X_train[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))


        #To visualize and save the images
        for i in range(1,len(eps)):
            epsilon = eps[i]
            adv_x = fgsm(x, predictions, eps=epsilon)
            eval_params = {'batch_size': FLAGS.batch_size}
            X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
            X_train_adv, = batch_eval(sess, [x], [adv_x], [X_train], args=eval_params)
            accOrigAd[i] = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                              args=eval_params)
            print('Test accuracy on Adversarial Samples with eps =    ' + str(epsilon) + '  is:   ' +str(accOrigAd[i]))

            if(FLAGS.nb_channels==1):
                adversarial = np.reshape(X_train_adv[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols))
            else:
                adversarial = np.reshape(X_train_adv[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))

            figures['im_'+str(epsilon)] = adversarial
           
            
            if not os.path.exists(os.path.join(dataFolder,'Train',str(epsilon))):
                os.makedirs(os.path.join(dataFolder,'Train',str(epsilon)))
            if not os.path.exists(os.path.join(dataFolder,'Test',str(epsilon))):
                os.makedirs(os.path.join(dataFolder,'Test',str(epsilon)))
            
            for j in range(0,len(classes)):
                if not os.path.join(dataFolder,'Train',str(epsilon),str(classes[j])):
                    os.makedirs(os.path.join(dataFolder,'Train',str(epsilon),str(classes[j])))
                if not os.path.join(dataFolder,'Test',str(epsilon),str(classes[j])):
                    os.makedirs(os.path.join(dataFolder,'Test',str(epsilon),str(classes[j])))

            for u in range(0,np.shape(X_train_adv)[0]):
                current_class = int(np.argmax(Y_train[u]))
                
                if(FLAGS.nb_channels==1):
                    adversarial = np.reshape(X_train_adv[u:(u+1)],(FLAGS.img_rows, FLAGS.img_cols))
                    scipy.misc.imsave(os.path.join(dataFolder,'Train',str(epsilon),str(classes[current_class]),str(u)+'.png'), adversarial)
                else:
                    adversarial = np.reshape(X_train_adv[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))
                    scipy.misc.imsave(os.path.join(dataFolder,'Train',str(epsilon),str(classes[current_class]),str(u)+'.png'), adversarial)

            for u in range(0,np.shape(X_test_adv)[0]):
                current_class = int(np.argmax(Y_test[u]))
                if(FLAGS.nb_channels==1):
                    adversarial = np.reshape(X_test_adv[u:(u+1)],(FLAGS.img_rows, FLAGS.img_cols))
                    scipy.misc.imsave(os.path.join(dataFolder,'Test',str(epsilon),str(classes[current_class]),str(u)+'.png'), adversarial)
                else:
                    adversarial = np.reshape(X_test_adv[u:(u+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))
                    scipy.misc.imsave(os.path.join(dataFolder,'Test',str(epsilon),str(classes[current_class]),str(u)+'.png'), adversarial)


        plot_figures(figures, 5, 10,dataFolder,'Epsilon.png',FLAGS.nb_channels, eps)
        print(accuracy)
        accuracy2 = model_eval(sess, x, y, predictions, X_train, Y_train, args=eval_params)
        print('training acc')
        print(accuracy2)



    if(Training == 1):
        # Train after augmenting examples separately for different epsilons.
        for i in range(1,len(eps)):
            
            epsilon = eps[i]
            
            adv_x = fgsm(x, predictions, eps=epsilon)
            eval_params = {'batch_size': FLAGS.batch_size}
            X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
            X_train_adv, = batch_eval(sess, [x], [adv_x], [X_train], args=eval_params)

            
            X_trainN = np.append(X_train,X_train_adv,axis=0)
            X_testN = np.append(X_test,X_test_adv,axis=0)
            Y_trainN = np.append(Y_train,Y_train,axis=0)
            Y_testN = np.append(Y_test,Y_test,axis=0)
            
            if(datacode == 1):
                    

                    model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 30, nb_classes)
                    label_smooth = .1
                    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
            elif(datacode == 2):
               
                model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 40, nb_classes)

            elif(datacode == 3):
               
                tf.set_random_seed(5)
                model = cnn_model(False, None, FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels, 50, nb_classes)
            # Train an MNIST model
            train_params = {
                    'nb_epochs': FLAGS.nb_epochs,
                    'batch_size': FLAGS.batch_size,
                    'learning_rate': FLAGS.learning_rate
            }

               
            eval_params = {'batch_size': FLAGS.batch_size}
            predictions = model(x)
            print('model training begins--------------------------------------------------------------------------------------------')
            model_train(sess, x, y, predictions, X_trainN, Y_trainN, evaluate=evaluate, args=train_params)
            print('model training ends----------------------------------------------------------------------------------------------')
            

            accAdTest[i] = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
            print('Adversarially trained model accuracy on Test Samples with eps =    ' + str(epsilon) + '  is:   ' +str(accAdTest[i]))

            accAdAd[i] = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                              args=eval_params)
            print('Adversarially trained model accuracy on Adversarial Samples with eps =    ' + str(epsilon) + '  is:   ' +str(accAdAd[i]))
            print('Original model accuracy on Adversarial Samples with eps =    ' + str(epsilon) + '  is:   ' +str(accOrigAd[i]))
            

        np.array(accAdAd).dump(open(os.path.join(dataFolder,'accAdAd.npy'), 'wb'))
        np.array(accAdTest).dump(open(os.path.join(dataFolder,'accAdTest.npy'), 'wb'))
        np.array(accOrigAd).dump(open(os.path.join(dataFolder,'accOrigAd.npy'), 'wb'))
        np.array(eps).dump(open(os.path.join(dataFolder,'eps.npy'), 'wb'))
        print("--- %s seconds ---" % (time.time() - start_time))

        # Create plots with pre-defined labels.
        # Alternatively, you can pass labels explicitly when calling `legend`.
        fig, ax = plt.subplots()
        ax.plot(eps, accAdTest, 'm--', label='Accuracy Adversarial Trained Model on Test Data')
        ax.plot(eps, accAdAd, 'b:', label='Accuracy Adversarial Trained Model on Advers Test Data')
        ax.plot(eps, accOrigAd, 'g', label='Accuracy original Model on Adversarial Test Data')


        # Now add the legend with some customizations.
        legend = ax.legend(loc='center right', shadow=True)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
                    label.set_fontsize('small')

        for label in legend.get_lines():
                    label.set_linewidth(1.5)  # the legend line width
            
        plt.show()
        fig.savefig(os.path.join(dataFolder,'Graph.png'))
        plt.close(fig)



    if(Testing == 1):


        # Evaluate the accuracy of the MNIST model on Transformation based augmented samples (for cars dataset)
        # dataTestingX, dataTestingY stores the samples 
        #model is the original model trained on original samples.

        #This function basically generates adversarial samples corresponding to transformation based augmented samples.
        
        accuracy = model_eval(sess, x, y, predictions, dataTestingX, dataTestingY, args=eval_params)
        print('Test accuracy on Augmented Samples with eps =    ' + str(0) + '  is:   ' +str(accuracy))
        sample_ind = 0

        number_of_im = len(eps)
        figures = {}
        if(FLAGS.nb_channels==1):
            figures['im_0']=np.reshape(X_train[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols))
        else:
            figures['im_0']=np.reshape(X_train[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))



        #To generate adversarial examples corresponding to different data set.
        for i in range(1,len(eps)):
            epsilon = eps[i]
            adv_x = fgsm(x, predictions, eps=epsilon)
            eval_params = {'batch_size': FLAGS.batch_size}
            X_test_adv, = batch_eval(sess, [x], [adv_x], [dataTestingX], args=eval_params)
            accOrigAd[i] = model_eval(sess, x, y, predictions, X_test_adv, dataTestingY,
                              args=eval_params)
            print('Test accuracy on Adversarial Samples with eps =    ' + str(epsilon) + '  is:   ' +str(accOrigAd[i]))

            if(FLAGS.nb_channels==1):
                adversarial = np.reshape(X_test_adv[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols))
            else:
                adversarial = np.reshape(X_test_adv[sample_ind:(sample_ind+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))

            figures['im_'+str(epsilon)] = adversarial
           
            

            os.makedirs(os.path.join(saveFolderTest,str(epsilon)))
            for j in range(0,len(classes)):
                os.makedirs(os.path.join(saveFolderTest,str(epsilon),str(classes[j])))

           
            for u in range(0,np.shape(X_test_adv)[0]):
                current_class = int(np.argmax(dataTestingY[u]))
                if(FLAGS.nb_channels==1):
                    adversarial = np.reshape(X_test_adv[u:(u+1)],(FLAGS.img_rows, FLAGS.img_cols))
                    scipy.misc.imsave(os.path.join(saveFolderTest,str(epsilon),str(classes[current_class]),str(u)+'.png'), adversarial)
                else:
                    adversarial = np.reshape(X_test_adv[u:(u+1)],(FLAGS.img_rows, FLAGS.img_cols,FLAGS.nb_channels))
                    scipy.misc.imsave(os.path.join(saveFolderTest,str(epsilon),str(classes[current_class]),str(u)+'.png'), adversarial)

        plot_figures(figures, 5, 10,dataFolder,'Epsilon.png',FLAGS.nb_channels, eps)     


if __name__ == '__main__':
    app.run()
