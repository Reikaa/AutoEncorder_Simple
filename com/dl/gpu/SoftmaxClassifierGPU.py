__author__ = 'Thushan Ganegedara'

from math import sqrt
from math import isnan,isinf
import numpy as np
from scipy import optimize
import numpy.linalg as LA
class SoftmaxClassifier(object):

    def sigmoid(self, x):
        return np.exp(x)


    def __init__(self, n_inputs, n_outputs, X, Y, W1=None, b1=None):

        self.X = X
        self.Y = Y

        #define global variables for n_inputs and n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs


        #generate random weights for W
        if W1 == None:
            val_range1 = [0, 2*sqrt(6.0/(n_inputs+n_outputs+1))]
            W1 = np.random.random_sample((n_outputs, n_inputs))*0.2
            self.W1 = W1


        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = np.random.random_sample((n_outputs,)) * 0.2
            self.b1 = b1


    def forward_pass(self, x, W1, b1):

        x_bias = np.concatenate((x,np.array([1])),axis=0)
        theta_mat = self.getThetaMatrix(W1,b1)

        top = np.exp(np.dot(theta_mat,x_bias))
        bottom = np.sum(top)

        return top/bottom

    def back_prop(self, iter=1000):
        init_val = self.packTheta(self.W1, self.b1)
        #err = optimize.check_grad(self.cost, self.cost_prime, init_val, self.X,self.Y)
        res = optimize.minimize(fun=self.cost, x0=init_val, args=(self.X,self.Y,0.0001), jac=self.cost_prime, method='L-BFGS-B', options={'maxiter':iter,'disp':True})
        self.W1, self.b1 = self.unpackTheta(res.x)

        #print ("Error (check_grad): %f" %err)



    def get_params(self):
        return self.W1,self.b1