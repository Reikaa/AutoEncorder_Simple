__author__ = 'Thushan Ganegedara'


import numpy as np
from theano import function, config, shared, sandbox
import theano.tensor as T

import numpy.linalg as LA
class SoftmaxClassifier(object):

    def sigmoid(self, x):
        return np.exp(x)


    def __init__(self, n_inputs, n_outputs, x=None, y=None, y_mat=None, W1=None, b1=None):

        self.x = x
        self.y = y
        self.y_mat = y_mat

        #define global variables for n_inputs and n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        #generate random weights for W
        if W1 == None:
            W1 = np.random.random_sample((n_inputs,n_outputs))*0.2
            self.W1 = shared(value=W1, name = 'W1', borrow=True)

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = np.random.random_sample((n_outputs,)) * 0.2
            self.b1 = shared(value=b1, name = 'b1', borrow=True)

        self.theta = [self.W1,self.b1]




    def forward_pass(self, input):

        a = T.nnet.softmax(T.dot(input,self.W1) + self.b1)
        return a

    def get_cost_and_weight_update(self, l_rate):

        a = self.forward_pass(self.x)

        cost = T.mean(0.5 * T.sum(T.sqr(a-self.y_mat), axis=1))

        #gparam is a list of elements where each correspond to a single item in theta
        #here for example, theta =[W1,b1] there for gparam = [W1_grad,b1_grad]
        gparams = T.grad(cost, wrt=self.theta)

        updates = [
            (param, param - l_rate*gparam)
            for param, gparam in zip(self.theta,gparams)]

        return cost, updates

    def get_params(self):
        return self.theta