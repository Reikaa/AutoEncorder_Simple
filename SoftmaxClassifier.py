__author__ = 'Thushan Ganegedara'

from math import sqrt
import numpy as np
from scipy import optimize
class SoftmaxClassifier(object):

    def sigmoid(self, x):
        return np.exp(x)


    def __init__(self, n_inputs, n_outputs, X, Y, W1=None, W2=None, b1=None, b2=None):

        self.X = X
        self.Y = Y

        #define global variables for n_inputs and n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.lam = 0.0001

        #generate random weights for W
        if W1 == None:
            val_range1 = [-sqrt(6.0/(n_inputs+n_outputs+1)), sqrt(6.0/(n_inputs+n_outputs+1))]
            W1 = val_range1[0] + np.random.random_sample((n_outputs, n_inputs))*2.0*val_range1[1]
            self.W1 = W1


        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = -0.01 + np.random.random_sample((n_outputs,)) * 0.02
            self.b1 = b1


    def forward_pass_for_one_case(self, x, W1, b1):

        z2 = np.dot(W1, x) + b1
        a2 = self.sigmoid(z2)

        return a2

    def packTheta(self, W1, b1):
        theta = np.concatenate((np.reshape(W1, (self.n_outputs*self.n_inputs,)), b1))
        return theta

    def unpackTheta(self, theta):
        sIdx = 0
        W1 = np.reshape(theta[sIdx:self.n_inputs*self.n_outputs], (self.n_outputs, self.n_inputs))
        sIdx = self.n_outputs*self.n_inputs
        b1 = np.reshape(theta[sIdx:sIdx+self.n_outputs],(self.n_outputs,))

        return W1, b1


    # cost calculate the cost you get given all the inputs feed forward through the network
    # at the moment I am using the squared error between the reconstructed and the input
    # theta is a vector formed by unrolling W1,b1,W2,b2 in to a single vector
    # Theta will be the input that the optimization method trying to optimize
    # make sure to send 5,4,9.. type of values for the 'labels', not the vectorized form
    def cost(self, theta,data,labels):

        W1, b1 = self.unpackTheta(theta)
        tot_err = 0.0
        size_data = data.shape[1]

        for idx in range(size_data):
            x = data[:, idx]
            a2 = self.forward_pass_for_one_case(x, W1, b1)

            err = -np.log(a2[labels[idx]]/np.sum(a2))
            tot_err += err

        tot_err = tot_err/size_data

        return tot_err

    # Cost prime is the gradient of the cost function.
    # In other words this is dC/dW in the delta rule (i.e. W = W - alpha*dC/dW)
    # make sure to send 5,4,9.. type of values for the 'labels', not the vectorized form
    def cost_prime(self,theta,data,labels):

        W1, b1 = self.unpackTheta(theta)

        d_W1 = np.zeros((self.n_hidden, self.n_inputs), dtype=np.float32)
        d_b1 = np.zeros((self.n_hidden,), dtype=np.float32)

        size_data = data.shape[1]
        for idx in range(size_data):
            x = data[:,idx]
            y = labels[idx]
            y_vec = [0.0] * self.n_outputs
            y_vec[y] = 1.0
            a2 = self.forward_pass_for_one_case(x, W1, b1)

            delta = y_vec - np.log(a2[y]/np.sum(a2))
            d_W1 = d_W1 + np.dot(delta[:, None], np.transpose(x[:, None]))
            d_b1 = d_b1 + delta

        d_W1 = ((1.0/size_data) * d_W1) + (self.lam * W1)
        d_b1 = (1.0/size_data) * d_b1

        return self.packTheta(d_W1, d_b1)

    def back_prop(self, iter=1000):
        init_val = self.packTheta(self.W1, self.b1)
        res = optimize.minimize(fun=self.cost, x0=init_val, args=(self.X,), jac=self.cost_prime, method='L-BFGS-B', options={'maxiter':iter,'disp':True})
        self.W1, self.b1 = self.unpackTheta(res.x)
