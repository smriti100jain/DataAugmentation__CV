Installations:

python 2.7
tensorflow 0.12 (with GPU support for fast computation)
keras with tensorflow backend. For this

Cleverhans toolbox ( https://github.com/tensorflow/cleverhans ) : 

git clone https://github.com/tensorflow/cleverhans
export PYTHONPATH="/path/to/cleverhans":$PYTHONPATH

require GPU support.


All adversarial data augmentation techniques tested for Image Classification

1. Fast Gradient Sign Method:

python fastgradientsignmethod.py

user inputs
-1. blackwhite 0: color
1
-Enter a dataset code 1:mnist,2:pasta,3:car,4:caraugmented
1
-Enter 1 if you want to train at each epsilon after augmentation (very time consuming) else just adversarial examples will be stored 
0
-Enter 1 if you want to test your model on some other data and new adversarial samples
0


Jacobian Saliency Map Value:

python jacobiansaliencymapapproach.py

user inputs
-1. blackwhite 0: color
1
-Enter a dataset code 1:mnist,2:pasta,3:car
1
-Press 1 if Adversarial Examples corresponding to Train Examples else 0
1


Conditional DCGAN:

MNISI:

cd dcgan
cd MNIST

to train generative model. models and samples are saved at respective folders
python train_cond_dcgan.py
[inspired from https://github.com/Newmu/dcgan_code for 28x28 images]

to load save generator and discriminator model and visualize the performance by looking at all the classes images.

python loadModelMnistCondGAN.py

[for MNIST, I already trained and saved models in '/models/']

-to save samples images in a folder:

python utilGAN2.py

For pasta_blackANDwhite and cars:

the model is changed to generate 200x200 images but the trained models are not saved due to confidentiality constraint of XRCE data sets

cd pasta_blackANDwhite

-to train:
python train_cond_dcgan.py

-to load trained model and evaluate performance by visualizing
python loadModelcond_dcgan.py

-to save sampled images for using as data augmentation
python utilsGAN_vs1.py

and similarly for cars dataset.



Info GAN

for MNIST:
--training:
python trainMNIST.py
--for sampling and saving images:
python genMNISTimages.py

for pasta (32x32) training
for pasta (128x128) training

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>