
import sys
import os

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


def mnistGANcond():
    """
    This example loads the 32x32 imagenet model used in the paper,
    generates 400 random samples, and sorts them according to the
    discriminator's probability of being real and renders them to
    the file samples.png
    """

    nc = 1
    npx = 28
    ngf = 64          # # of gen filters in first conv layer
    ndf = 128
    ny = 10           # # of classes

    nz = 100          # # of dim for Z
    k = 1             # # of discrim updates for each gen update
    l2 = 2.5e-5       # l2 weight decay
    b1 = 0.5          # momentum term of adam
    nc = 1            # # of channels in image
    ny = 10           # # of classes
    nbatch = 128      # # of examples in batch
    npx = 28          # # of pixels width/height of images
    nz = 100          # # of dim for Z
    ngfc = 1024       # # of gen units for fully connected layers
    ndfc = 1024       # # of discrim units for fully connected layers
    ngf = 64          # # of gen filters in first conv layer
    ndf = 64          # # of discrim filters in first conv layer
    nx = npx*npx*nc   # # of dimensions in X
    niter = 100       # # of iter at starting learning rate
    niter_decay = 100 # # of iter to linearly decay learning rate to zero
    lr = 0.0002    


    relu = activations.Rectify()
    sigmoid = activations.Sigmoid()
    lrelu = activations.LeakyRectify()
    tanh = activations.Tanh()

    model_path = 'dcgan_code-master/mnist/models/cond_dcgan/'
    gen_params = [sharedX(p) for p in joblib.load(model_path+'200_gen_params.jl')]
    discrim_params = [sharedX(p) for p in joblib.load(model_path+'200_discrim_params.jl')]

    def gen(Z, Y, w, w2, w3, wx):
        yb = Y.dimshuffle(0, 1, 'x', 'x')
        Z = T.concatenate([Z, Y], axis=1)
        h = relu(batchnorm(T.dot(Z, w)))
        h = T.concatenate([h, Y], axis=1)
        h2 = relu(batchnorm(T.dot(h, w2)))
        h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
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

    Z = T.matrix()
    X = T.tensor4()
    Y = T.matrix()

    gX = gen(Z,Y, *gen_params)
    dX = discrim(X,Y, *discrim_params)

    _gen = theano.function([Z,Y], gX)
    _discrim = theano.function([X,Y], dX)

    sample_zmb = floatX(np_rng.uniform(-1., 1., size=(200, nz)))
    sample_ymb = floatX(OneHot(np.asarray([[i for _ in range(20)] for i in range(10)]).flatten(), ny))
    samples = _gen(sample_zmb,sample_ymb)
    scores = _discrim(samples,sample_ymb)
    print(scores[1:10])
    sort = np.argsort(scores.flatten())[::-1]
    samples = samples[sort]
    print(np.shape(inverse_transform(samples)))
    print(min(scores))
    print(max(scores))

    color_grid_vis(inverse_transform(samples), (20, 20), 'samples.png')

    return inverse_transform(samples), sample_ymb