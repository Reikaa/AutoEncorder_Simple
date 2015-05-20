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

    def packTheta(self, W1, b1):
        theta = np.concatenate((np.reshape(W1, (self.n_outputs*self.n_inputs,)), b1))
        return theta

    def unpackTheta(self, theta):
        sIdx = 0
        W1 = np.reshape(theta[sIdx:self.n_inputs*self.n_outputs], (self.n_outputs, self.n_inputs))
        sIdx = self.n_outputs*self.n_inputs
        b1 = np.reshape(theta[sIdx:sIdx+self.n_outputs],(self.n_outputs,))

        return W1, b1

    def getThetaMatrix(self, W1, b1):
        theta_mat =  np.append(W1,b1[:,None],axis=1)
        return theta_mat

    def getWeightsAndBias(self, theta_mat):
        W1 = theta_mat[:,0:theta_mat.shape[1]-1]
        b1 = theta_mat[:,theta_mat.shape[1]-1]
        return W1,b1

    # cost calculate the cost you get given all the inputs feed forward through the network
    # at the moment I am using the squared error between the reconstructed and the input
    # theta is a vector formed by unrolling W1,b1,W2,b2 in to a single vector
    # Theta will be the input that the optimization method trying to optimize
    # make sure to send 5,4,9.. type of values for the 'labels', not the vectorized form
    def cost(self, theta,data,labels, lam=0.5):

        W1, b1 = self.unpackTheta(theta)
        tot_err = 0.0
        size_data = data.shape[1]

        test = np.sum(data,axis=1)
        for idx in range(size_data):
            x = data[:, idx]
            x_bias = np.concatenate((x,np.array([1])),axis=0)

            y = labels[idx]

            theta_mat = self.getThetaMatrix(W1,b1)
            theta_k = theta_mat[y,:]
            tmp = np.dot(theta_k,x_bias)
            log_top = np.exp(tmp)
            tmp3 = np.exp(np.dot(theta_mat,x_bias))
            log_bottom = np.sum(tmp3)

            if log_bottom == 0. or isnan(log_bottom) or isinf(log_bottom):
                print "test"

            tmp2 = np.log(log_top/log_bottom)
            err = -np.sum(tmp2)

            tot_err += err

        tot_err = tot_err/size_data + (lam/2)*np.sum(np.sum(theta_mat**2))

        return tot_err

    # Cost prime is the gradient of the cost function.
    # In other words this is dC/dW in the delta rule (i.e. W = W - alpha*dC/dW)
    # make sure to send 5,4,9.. type of values for the 'labels', not the vectorized form
    def cost_prime(self, theta, data, labels, lam=0.5):

        W1, b1 = self.unpackTheta(theta)

        d_theta = np.zeros((self.n_outputs, self.n_inputs+1), dtype=np.float32)

        size_data = data.shape[1]
        for idx in range(size_data):
            x = data[:,idx]
            x_bias = np.concatenate((x,np.array([1])),axis=0)

            y = labels[idx]
            y_vec = [0.0] * self.n_outputs
            y_vec[y] = 1.0

            theta_mat = self.getThetaMatrix(W1, b1)

            W1,b1 = self.getWeightsAndBias(theta_mat)

            topDevBottom = self.forward_pass(x,W1,b1)[y]

            delta = np.asarray(y_vec,dtype=np.float32)
            delta[y,None] = delta[y,None]-topDevBottom

            tmp_arr = np.dot(delta[:, None], np.transpose(x_bias[:, None]))
            d_theta = d_theta + tmp_arr

        d_theta = -((1.0/size_data) * d_theta) + lam*theta_mat

        return np.reshape(d_theta,(self.n_outputs*(self.n_inputs+1),))

    def back_prop(self, iter=1000):
        init_val = self.packTheta(self.W1, self.b1)
        #err = optimize.check_grad(self.cost, self.cost_prime, init_val, self.X,self.Y)
        res = optimize.minimize(fun=self.cost, x0=init_val, args=(self.X,self.Y,0.0001), jac=self.cost_prime, method='L-BFGS-B', options={'maxiter':iter,'disp':True})
        self.W1, self.b1 = self.unpackTheta(res.x)

        #print ("Error (check_grad): %f" %err)

    def back_prop_man(self, iter=1000, alpha=0.5, lam =0.0001):

        theta_mat = self.getThetaMatrix(self.W1, self.b1)

        for i in range(iter):

            d_theta = np.zeros((self.n_outputs, self.n_inputs+1), dtype=np.float32)
            size_data = self.X.shape[1]
            tot_err = 0.0

            for idx in range(size_data):
                x = self.X[:,idx]
                x_bias = np.concatenate((x,np.array([1])),axis=0)

                y = self.Y[idx]
                y_vec = [0.0] * self.n_outputs
                y_vec[y] = 1.0

                W1,b1 = self.getWeightsAndBias(theta_mat)
                topDevBottom = self.forward_pass(x,W1,b1)

                delta = np.asarray(y_vec,dtype=np.float32)
                delta = delta-topDevBottom
                err = LA.norm(delta)
                tot_err = tot_err + err

                tmp_arr = np.dot(delta[:, None], np.transpose(x_bias[:, None]))
                d_theta = d_theta + tmp_arr

            d_theta = -((1.0/size_data) * d_theta) + (lam*theta_mat)
            tot_err = (1.0/size_data)*tot_err
            print ("Iteration: %i, Error: %f"%(i,tot_err))
            theta_mat = theta_mat - (alpha*d_theta)


        self.W1 = theta_mat[:,0:theta_mat.shape[1]-1]
        self.b1 = theta_mat[:,theta_mat.shape[1]-1]


    def get_params(self):
        return self.W1,self.b1