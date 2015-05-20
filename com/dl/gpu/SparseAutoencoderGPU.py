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

class SparseAutoencoder(object):


    def kl_diverg(self,rho,rho_nh):
        return rho*np.log(rho/rho_nh) + (1-rho)*np.log((1-rho)/(1-rho_nh))

    def d_kl_diverg(self,rho,rho_nh):
        return -(rho/rho_nh)+((1-rho)/(1-rho_nh))


    def dsigmoid(self, y):
        return y * (1.0 - y)
        #return 1.0-y**2

    #the value specified in the argument for each variable is the default value
    #__init__ is called when the constructor of an object is called (i.e. created an object)

    #by reducing number of hidden from 400 -> 75 and hidden2 200 -> 25 got an error reduction of 540+ -> 387 (for numbers dataset)
    def __init__(self, n_inputs, n_hidden, x=None, W1=None, W2=None, b1=None, b2=None):
        self.x = x

        #define global variables for n_inputs and n_hidden
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs

        self.lam = 0.0001
        self.beta = 0.1
        self.rho = 0.3

        self.rho_nh = np.zeros((self.n_hidden,),dtype=np.float32)

        #generate random weights for W
        if W1 == None:
            val_range1 = [-math.sqrt(6.0/(n_inputs+n_hidden+1)), math.sqrt(6.0/(n_inputs+n_hidden+1))]
            W1 = val_range1[0] + np.random.random_sample((n_hidden, n_inputs))*2.0*val_range1[1]
            W1 = np.asarray(W1,dtype=config.floatX)
            self.W1 = shared(value=W1, name='W1', borrow=True)

        if W2 == None:
            val_range2 = [-math.sqrt(6.0/(self.n_outputs+n_hidden+1)), math.sqrt(6.0/(self.n_outputs+n_hidden+1))]
            W2 = val_range2[0] + np.random.random_sample((self.n_outputs, n_hidden))*2.0*val_range2[1]
            W2 = np.asarray(W2,dtype=config.floatX)
            self.W2 = shared(value=W2, name='W2', borrow=True)

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = -0.01 + np.random.random_sample((n_hidden,)) * 0.02
            self.b1 = shared(value=np.asarray(b1, dtype=config.floatX), name='b1', borrow=True)

        if b2 == None:
            b2 = -0.02 + np.random.random_sample((self.n_outputs,)) * 0.04
            self.b2 = shared(value=np.asarray(b2, dtype=config.floatX), name='b2', borrow=True)

        self.theta = [self.W1,self.b1,self.W2,self.b2]

    def forward_pass(self, x):

        a2 = T.nnet.sigmoid(T.dot(self.W1, x)+self.b1)
        a3 = T.nnet.sigmoid(T.dot(self.W2, a2)+self.b2)
        return a2, a3

    def packTheta(self, W1, b1, W2, b2):
        theta_p1 = T.concatenate((T.reshape(W1, (self.n_hidden*self.n_inputs,)), b1))
        theta_p2 = T.concatenate((T.reshape(W2, (self.n_outputs*self.n_hidden,)), b2))
        theta = T.concatenate((theta_p1,theta_p2))
        return theta

    def unpackTheta(self, theta):
        sIdx = 0
        W1 = T.reshape(theta[sIdx:self.n_inputs*self.n_hidden], (self.n_hidden, self.n_inputs))
        sIdx = self.n_hidden*self.n_inputs
        b1 = T.reshape(theta[sIdx:sIdx+self.n_hidden],(self.n_hidden,))
        sIdx = sIdx + self.n_hidden
        W2 = T.reshape(theta[sIdx:sIdx + self.n_outputs*self.n_hidden],(self.n_outputs,self.n_hidden))
        sIdx = sIdx + self.n_outputs*self.n_hidden
        b2 = T.reshape(theta[sIdx:],(self.n_outputs,))

        return W1, b1, W2, b2

    # cost calculate the cost you get given all the inputs feed forward through the network
    # at the moment I am using the squared error between the reconstructed and the input
    # theta is a vector formed by unrolling W1,b1,W2,b2 in to a single vector
    # Theta will be the input that the optimization method trying to optimize
    def get_cost_and_weight_update(self, l_rate):

        a2,a3 = self.forward_pass(self.x)

        #cost = - T.sum(self.x * T.log(a3) + (1 - self.x) * T.log(1 - a3), axis=1)
        cost = T.mean(0.5 * T.sum(T.sqr(a3-self.x), axis=1))

        gparams = T.grad(cost, self.theta)

        #zip is used to iterate over two lists in parallel. This says, give the updated values for
        #param and gparam (param and param - l_rate*gparam respectively) while enumerating
        #params and gparams in parallel
        updates = [
            (param, param - l_rate*gparam)
            for param, gparam in zip(self.theta, gparams)
        ]

        return cost, updates

        ''' Cost Function
        W1, b1, W2, b2 = self.unpackTheta(theta)
        tot_sqr_err = 0.0
        size_data = data.shape[1]

        self.rho_nh = np.ones((self.n_hidden,),dtype=np.float32)*0.001

        for idx in range(size_data):
            x = data[:, idx]
            a2, a3 = self.forward_pass_for_one_case(x, W1, b1, W2, b2)
            self.rho_nh = self.rho_nh + a2
            sqr_err = 0.5 * np.sum((a3-x)**2) + (self.lam/2)*np.sum(np.sum(W1**2,axis=1),axis=0) + (self.lam/2)*np.sum(np.sum(W2**2,axis=1), axis=0)
            tot_sqr_err += sqr_err

        self.rho_nh = self.rho_nh/size_data
        tot_err = tot_sqr_err/size_data + self.beta*np.sum(self.kl_diverg(self.rho,self.rho_nh))

        return tot_err'''


#this calls the __init__ method automatically
#dA = SparseAutoEncoder()
#dA.load_data()
#dA.back_prop()
#dA.test_back_prop_with_diff_grad_checks()

#dA.save_hidden()
#dA.save_reconstructed()
