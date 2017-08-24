
import sys
import os
sys.path.append('..')
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import costs
from lib import inits
from lib import updates
from lib import activations
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX, intX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from sklearn.externals import joblib
import scipy
from scipy import misc
import pickle
from load import pastaBlackWhite

trX, vaX, teX, trY, vaY, teY = pastaBlackWhite()


lr = 0.0002    


k = 2             # # of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
ny = 13           # # of classes
nbatch = 30      # # of examples in batch
npx = 200          # # of pixels width/height of images
nz = 200          # # of dim for Z
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X

niter = 3000       # # of iter at starting learning rate
niter_decay = 3000 # # of iter to linearly decay learning rate to zero
temp = npx/4


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()

model_path = 'models/cond_dcgan/'
gen_params = [sharedX(p) for p in joblib.load(model_path+'5999_gen_params.jl')]
discrim_params = [sharedX(p) for p in joblib.load(model_path+'5999_discrim_params.jl')]

def gen(Z, Y, w, w2, w3, wx):
        yb = Y.dimshuffle(0, 1, 'x', 'x')
        Z = T.concatenate([Z, Y], axis=1)
        h = relu(batchnorm(T.dot(Z, w)))
        h = T.concatenate([h, Y], axis=1)
        h2 = relu(batchnorm(T.dot(h, w2)))
        h2 = h2.reshape((h2.shape[0], ngf*2, temp, temp))
        h2 = conv_cond_concat(h2, yb)
        h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
        h3 = conv_cond_concat(h3, yb)
        x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
        return x

def discrim(X, Y, w, w2, w3, wy):
        yb = Y.dimshuffle(0, 1, 'x', 'x')
        X = conv_cond_concat(X, yb)
        h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
        h = conv_cond_concat(h, yb)
        h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
        h2 = T.flatten(h2, 2)
        h2 = T.concatenate([h2, Y], axis=1)
        h3 = lrelu(batchnorm(T.dot(h2, w3)))
        h3 = T.concatenate([h3, Y], axis=1)
        y = sigmoid(T.dot(h3, wy))
        return y

def inverse_transform(X):
        X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
        return X

def transform(X):
    return (floatX(X)/255.).reshape(-1, nc, npx, npx)

Z = T.matrix()
X = T.tensor4()
Y = T.matrix()

gX = gen(Z,Y, *gen_params)
dX = discrim(X,Y, *discrim_params)

_gen = theano.function([Z,Y], gX)
_discrim = theano.function([X,Y], dX)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(10*ny, nz)))
sample_ymb = floatX(OneHot(np.asarray([[i for _ in range(10)] for i in range(ny)]).flatten(), ny))
samples = _gen(sample_zmb,sample_ymb)

scores = _discrim(samples,sample_ymb)
color_grid_vis(inverse_transform(samples), (ny,10), 'samples.png')



for i in range(ny):
    Z = T.matrix()  
    X = T.tensor4()
    Y = T.matrix()

    gX = gen(Z,Y, *gen_params)
    dX = discrim(X,Y, *discrim_params)

    _gen = theano.function([Z,Y], gX)
    _discrim = theano.function([X,Y], dX)

    a = transform(teX)
    tempY = np.zeros(np.shape(teY))+i
    tempY = tempY.astype(int)
    y = floatX(OneHot(tempY.flatten(),ny))
    scores = _discrim(a,y)
    
    if(i==0):
        Prob =scores
    else:
        Prob = np.hstack((Prob,scores))
    print(scores)
predictedClass = np.argmax(Prob,axis=1)
print(np.shape(predictedClass))
a = np.where(predictedClass == teY)
print(np.shape(a))


c = 0
for i in range(np.shape(teX)[0]):
    Image = transform(teX[i])
    
    actualClass = teY[i]
    Prob = np.zeros(ny)

    for j in range(ny):
        Y = floatX(OneHot(np.asarray(j).flatten(),ny))
        score = _discrim(Image,Y)
        Prob[j] = score
    print(Prob)    
    predictedClass =  np.argmax(Prob)
    print(predictedClass)
    print('...............................................................')
    if(actualClass == predictedClass):
        c = c+1

print(c)
print(np.shape(teX)[0])

