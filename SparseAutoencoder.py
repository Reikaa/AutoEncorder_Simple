__author__ = 'Thushan Ganegedara'


import numpy as np
import math
import os
from scipy import optimize
from scipy import misc
from numpy import linalg as LA
from PIL import Image

class SparseAutoencoder(object):


    def kl_diverg(self,rho,rho_nh):
        if 0. in rho_nh:
            print "Zero detected"
        return rho*np.log(rho/rho_nh) + (1-rho)*np.log((1-rho)/(1-rho_nh))

    def d_kl_diverg(self,rho,rho_nh):
        if 0. in rho_nh:
            print "Zero detected"
        return -(rho/rho_nh)+((1-rho)/(1-rho_nh))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
        #return math.tanh(x)

    def dsigmoid(self, y):
        return y * (1.0 - y)
        #return 1.0-y**2

    #the value specified in the argument for each variable is the default value
    #__init__ is called when the constructor of an object is called (i.e. created an object)

    #by reducing number of hidden from 400 -> 75 and hidden2 200 -> 25 got an error reduction of 540+ -> 387 (for numbers dataset)
    def __init__(self, n_inputs, n_hidden, X, W1=None, W2=None, b1=None, b2=None, m_batch_size=30):
        self.X = X

        #define global variables for n_inputs and n_hidden
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs

        self.m_batch_size = m_batch_size
        self.min_batch_X = np.zeros((n_inputs, m_batch_size), dtype=np.float32)

        self.lam = 0.0001
        self.beta = 0.5
        self.rho = 0.25

        self.rho_nh = np.zeros((self.n_hidden,),dtype=np.float32)

        #generate random weights for W
        if W1 == None:
            val_range1 = [-math.sqrt(6.0/(n_inputs+n_hidden+1)), math.sqrt(6.0/(n_inputs+n_hidden+1))]
            W1 = val_range1[0] + np.random.random_sample((n_hidden, n_inputs))*2.0*val_range1[1]
            self.W1 = W1

        if W2 == None:
            val_range2 = [-math.sqrt(24.0/(self.n_outputs+n_hidden+1)), math.sqrt(24.0/(self.n_outputs+n_hidden+1))]
            W2 = val_range2[0] + np.random.random_sample((self.n_outputs, n_hidden))*2.0*val_range2[1]
            self.W2 = W2

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = -0.01 + np.random.random_sample((n_hidden,)) * 0.02
            self.b1 = b1

        if b2 == None:
            b2 = -0.02 + np.random.random_sample((self.n_outputs,)) * 0.04
            self.b2 = b2


    def load_data(self):

        dir_name = "Data"
        for i in range(1, 41):
            file_name = "\\image_"+str(i)+".jpg"
            img = misc.imread(dir_name+file_name)
            imgVec = np.reshape(img, (810, 1))
            self.X[:, i-1] = imgVec[:, 0]

        self.X = self.X/255.0
        #mean_X = np.mean(self.X, axis=0)
        #for i in range(self.X.shape[1]):
        #    self.X[:,i] = self.X[:,i]-mean_X[i,]

    def renew_dataset(self):
        sel_idx = []
        tmp_data = np.zeros((self.X.shape[0],self.m_batch_size),dtype=np.float32)
        for dIdx in range(self.m_batch_size):
            while True:
                tmp_idx = np.random.randint(self.X.shape[1])
                if tmp_idx not in sel_idx:
                    sel_idx.append(tmp_idx)
                    tmp_data[:, dIdx] = self.X[:, tmp_idx]
                    break
        return tmp_data

    def forward_pass_for_one_case(self, x, W1, b1, W2, b2):

        z2 = np.dot(W1, x) + b1
        a2 = self.sigmoid(z2)

        z3 = np.dot(W2, a2) + b2
        a3 = self.sigmoid(z3)

        return a2, a3

    def packTheta(self, W1, b1, W2, b2):
        theta_p1 = np.concatenate((np.reshape(W1, (self.n_hidden*self.n_inputs,)), b1))
        theta_p2 = np.concatenate((np.reshape(W2, (self.n_outputs*self.n_hidden,)), b2))
        theta = np.concatenate((theta_p1,theta_p2))
        return theta

    def unpackTheta(self, theta):
        sIdx = 0
        W1 = np.reshape(theta[sIdx:self.n_inputs*self.n_hidden], (self.n_hidden, self.n_inputs))
        sIdx = self.n_hidden*self.n_inputs
        b1 = np.reshape(theta[sIdx:sIdx+self.n_hidden],(self.n_hidden,))
        sIdx = sIdx + self.n_hidden
        W2 = np.reshape(theta[sIdx:sIdx + self.n_outputs*self.n_hidden],(self.n_outputs,self.n_hidden))
        sIdx = sIdx + self.n_outputs*self.n_hidden
        b2 = np.reshape(theta[sIdx:],(self.n_outputs,))

        return W1, b1, W2, b2

    # cost calculate the cost you get given all the inputs feed forward through the network
    # at the moment I am using the squared error between the reconstructed and the input
    # theta is a vector formed by unrolling W1,b1,W2,b2 in to a single vector
    # Theta will be the input that the optimization method trying to optimize
    def cost(self, theta,data):

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

        return tot_err

    # Cost prime is the gradient of the cost function.
    # In other words this is dC/dW in the delta rule (i.e. W = W - alpha*dC/dW)
    def cost_prime(self,theta,data):

        #if all the elements of rho_nh are zero, initialize the array with the given value
        if all(rho_j==0. for rho_j in self.rho_nh):
            self.rho_nh = np.ones((self.n_hidden,), dtype=np.float32)*0.001

        W1, b1, W2, b2 = self.unpackTheta(theta)

        d_W1 = np.zeros((self.n_hidden, self.n_inputs), dtype=np.float32)
        d_b1 = np.zeros((self.n_hidden,), dtype=np.float32)
        d_W2 = np.zeros((self.n_outputs, self.n_hidden), dtype=np.float32)
        d_b2 = np.zeros((self.n_outputs, ), dtype=np.float32)

        size_data = data.shape[1]
        for idx in range(size_data):
            x = data[:,idx]
            a2, a3 = self.forward_pass_for_one_case(x, W1, b1, W2, b2)

            delta3 = -(x - a3) * self.dsigmoid(a3)
            delta2 = (np.dot(np.transpose(W2), delta3)+self.beta*self.d_kl_diverg(self.rho,self.rho_nh)) * self.dsigmoid(a2)

            d_W2 = d_W2 + np.dot(delta3[:, None], np.transpose(a2[:, None]))
            d_b2 = d_b2 + delta3

            d_W1 = d_W1 + np.dot(delta2[:, None], np.transpose(x[:, None]))
            d_b1 = d_b1 + delta2

        d_W2 = ((1.0/size_data) * d_W2) + (self.lam * W2)
        d_b2 = (1.0/size_data) * d_b2
        d_W1 = ((1.0/size_data) * d_W1) + (self.lam * W1)
        d_b1 = (1.0/size_data) * d_b1

        return self.packTheta(d_W1, d_b1, d_W2, d_b2)

    # back_propagation method uses Scipy optimize.minimize method to optimize Theta
    # fun - Cost function, x0 - initial Theta value, jac - gradient of cost function, method - optimization technique
    ''' Values shown by optimize.minimize when 'disp' = true
    # Tit   = total number of iterations
    # Tnf   = total number of function evaluations
    # Tnint = total number of segments explored during Cauchy searches
    # Skip  = number of BFGS updates skipped
    # Nact  = number of active bounds at final generalized Cauchy point
    # Projg = norm of the final projected gradient
    # F     = final function value '''
    def back_prop_with_SGD(self, iter=1500):
        #m_b_data = self.renew_dataset()
        init_val = self.packTheta(self.W1,self.b1,self.W2,self.b2)
        res = optimize.minimize(fun=self.cost, x0=init_val, args=(self.renew_dataset(),), jac=self.cost_prime, method='L-BFGS-B', options={'maxiter':iter,'disp':True})
        #err = optimize.check_grad(func=self.cost, x0=args, grad=self.cost_prime)
        #print err
        self.W1, self.b1, self.W2, self.b2 = self.unpackTheta(res.x)

    def back_prop(self, iter=1000):
        init_val = self.packTheta(self.W1, self.b1, self.W2, self.b2)
        res = optimize.minimize(fun=self.cost, x0=init_val, args=(self.X,), jac=self.cost_prime, method='L-BFGS-B', options={'maxiter':iter,'disp':True})
        self.W1, self.b1, self.W2, self.b2 = self.unpackTheta(res.x)


    def mkdir_if_not_exist(self, name):
        if not os.path.exists(name):
            os.makedirs(name)

    def save_hidden(self):
        h_dir = "Hidden"
        self.mkdir_if_not_exist(h_dir)
        for i in range(self.n_hidden):
            hImg = (self.W1[i,:]/LA.norm(self.W1[i, :]))*1000.0
            img = Image.fromarray(np.reshape(hImg, (27, 30))).convert('LA')
            img.save(h_dir + "\\" + 'hImg'+str(i)+'.png')

    #save reconstructed images
    def save_reconstructed(self):
        i_dir = "Reconstructed"
        self.mkdir_if_not_exist(i_dir)

        print ("Reconstructing the Inputs ...")
        for i in range(0, 40):
            #hImg = np.zeros((810,), dtype=np.int32)
            x = self.X[:, i]
            a2, a3 = self.forward_pass_for_one_case(x, self.W1, self.b1, self.W2, self.b2)
            if i > 0:
                rec_err = LA.norm(a3-x)*255.0
                print ("Reconstruction Error for image %i is %f" % (i+1, rec_err))
            rec_vec = a3*255.0
            rec_img = np.reshape(rec_vec, (27, 30))

            img = Image.fromarray(rec_img).convert('LA')
            img.save(i_dir + '\\recImg'+str(i+1)+'.png')

    def get_params(self):
        return self.W1,self.b1,self.W2,self.b2


#this calls the __init__ method automatically
#dA = SparseAutoEncoder()
#dA.load_data()
#dA.back_prop()
#dA.test_back_prop_with_diff_grad_checks()

#dA.save_hidden()
#dA.save_reconstructed()
