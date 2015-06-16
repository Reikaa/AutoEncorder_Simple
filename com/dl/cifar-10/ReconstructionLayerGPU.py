__author__ = 'Thushan Ganegedara'


import numpy as np
import math
import os
from scipy import optimize
from scipy import misc
from numpy import linalg as LA
from PIL import Image
from theano import function, config, shared, sandbox
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class ReconstructionLayer(object):

    #the value specified in the argument for each variable is the default value
    #__init__ is called when the constructor of an object is called (i.e. created an object)

    #by reducing number of hidden from 400 -> 75 and hidden2 200 -> 25 got an error reduction of 540+ -> 387 (for numbers dataset)
    def __init__(self, n_inputs, n_outputs, input=None, W=None, W2=None, b=None, b2=None):

        if input is None:
            self.input = T.dmatrix('input')
        else:
            self.input = input

        #define global variables for n_inputs and n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        numpy_rng = np.random.RandomState(89677)
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        #generate random weights for W
        if W == None:
            val_range1 = [-math.sqrt(6.0/(n_inputs+n_outputs+1)), math.sqrt(6.0/(n_inputs+n_outputs+1))]
            W = val_range1[0] + np.random.random_sample((n_inputs,n_outputs))*2.0*val_range1[1]
            W = np.asarray(W,dtype=config.floatX)
            self.W = shared(value=W, name='W', borrow=True)


        #by introducing *0.05 to b initialization got an error dropoff from 360 -> 280
        if b == None:
            b = -0.01 + np.random.random_sample((n_outputs,)) * 0.02
            self.b = shared(value=np.asarray(b, dtype=config.floatX), name='b', borrow=True)

        self.a = self.forward_pass(input=input,p=0.5,pre_training=False,dropout=False)
        self.theta = [self.W,self.b]

    def forward_pass(self,input=None,p=0.5,pre_training=False,dropout=False):

        if dropout:
            srng = T.shared_randomstreams.RandomStreams(np.random.randint(999999))
            mask = srng.binomial(n=1, p=1-p, size=(self.n_inputs,))

            if pre_training:
                input_tilda = input * mask
                a = T.nnet.sigmoid(T.dot(input_tilda,self.W) + self.b)
            else:
                a = T.nnet.sigmoid(T.dot(input,self.W*p) + self.b)
        else:
            a = T.nnet.sigmoid(T.dot(input,self.W) + self.b)

        return a


    def get_corrupted_input(self,input,corruption_level=0.3):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=config.floatX) * input


    def get_finetune_cost(self,input,lam,dropout=True):

        L = - T.sum(input * T.log(self.a) + (1 - input) * T.log(1 - self.a), axis=1)
        #L = 0.5 * T.sum(T.sqr(self.a-input), axis=1)
        cost = T.mean(L)

        return cost

    def get_error(self, input, dropout=True):
        L = 0.5 * T.sum(T.sqr(self.a-input), axis=1)
        return T.mean(L)

    def get_params(self):
        return [self.W, self.b]

    def get_hidden_act(self):
        return T.nnet.sigmoid(T.dot(self.input,self.W) + self.b)


