__author__ = 'Thushan Ganegedara'


import numpy as np
from theano import function, config, shared, sandbox
import theano.tensor as T

import numpy.linalg as LA
class SoftmaxClassifier(object):

    def sigmoid(self, x):
        return np.exp(x)


    def __init__(self, n_inputs, n_outputs, x=None, y=None, W1=None, b1=None):

        self.x = x
        self.y = y

        #define global variables for n_inputs and n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        #generate random weights for W
        if W1 == None:
            W1 = np.random.random_sample((n_inputs,n_outputs))*0.2
            self.W1 = shared(value=W1, name='W1', borrow=True)

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = np.random.random_sample((n_outputs,)) * 0.2
            self.b1 = shared(value=b1, name='b1', borrow=True)

        #Remember! These are symbolic experessions.
        self.a = self.forward_pass()
        self.pred = T.argmax(self.a, axis=1)

        self.theta = [self.W1,self.b1]


    def forward_pass(self):
        a = T.nnet.softmax(T.dot(self.x, self.W1) + self.b1)
        return a

    def get_cost(self):

        #The previous cost function is faulty. It's causing the error to go up
        #instead of reducing
        cost = -T.mean(T.log(self.a[T.arange(self.y.shape[0]), self.y]))
        return cost
        #return -T.mean(T.log(self.a)[T.arange(self.y.shape[0]), self.y])

    def get_params(self):
        return self.theta

    def get_error(self,y):
        return T.mean(T.neq(self.pred, y))