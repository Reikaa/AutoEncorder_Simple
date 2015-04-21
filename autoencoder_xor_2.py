__author__ = 'Thushan Ganegedara'

import numpy as np
import math
from scipy import misc
from numpy import linalg as LA
from PIL import Image

class SimpleAutoEncoder(object):


    def sigmoid(self, x):
        #return 1.0 / (1.0 + np.exp(-x))
        return math.tanh(x)

    def dsigmoid(self, y):
        #return y*(1.0-y)
        return 1.0 - y**2

    #the value specified in the argument for each variable is the default value
    #__init__ is called when the constructor of an object is called (i.e. created an object)
    def __init__(self, n_inputs=2, n_hidden=2, W1=None, W2=None, b1=None, b2=None):

        #define global variables for n_inputs and n_hidden
        self.n_h = n_hidden
        self.n_i = n_inputs
        self.n_o = 1

        #generate random weights for W
        if W1 == None:
            W1 = -0.2 + np.random.random_sample((n_hidden, n_inputs))*(0.2 - (-0.2))
            self.W1 = W1

        if W2 == None:
            W2 = -2.0+ np.random.random_sample((self.n_o, n_hidden))*(2.0 - (-2.0))
            self.W2 = W2

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = -0.20 + np.random.random_sample((n_hidden,)) * (0.2 - (-0.2))
            self.b1 = b1

        if b2 == None:
            b2 = -2.0 + np.random.random_sample((self.n_o,)) *(2.0 - (-2.0))
            self.b2 = b2



    def load_data(self):

        self.X = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.X = np.transpose(self.X)
        self.Y = np.array([0,1,1,0])


    def forward_pass_for_one_case(self, x):

        z2 = [0.0] * self.n_h
        a2 = [0.0] * self.n_h
        for j in range(self.n_h):
            sum = 0.0
            for i in range(self.n_i):
                sum = sum + (self.W1[j][i] * x[i])
            z2[j] = sum + self.b1[j]
            a2[j] = self.sigmoid(z2[j])

        z3 = [0.0] * self.n_o
        a3 = [0.0] * self.n_o

        for k in range(self.n_o):
            sum = 0.0
            for j in range(self.n_h):
                sum = sum + (self.W2[k][j]*a2[j])
            z3[k] = sum + self.b2[k]
            a3[k] = self.sigmoid(z3[k])

        return a2, a3

    def back_prop(self, iter=1000, alpha=0.3, w_decay=0.05):

        '''
            x -> W1 -> h -> W2 -> o
            W1 is a n_hidden x n_input matrix
            W2 is a n_output x n_hidden matrix
            i - iterating inputs
            j - iterating hidden
            k - iterating outputs
        '''

         #gradient descent
        delta_W1 = np.zeros((self.n_h,self.n_i),dtype=np.float32)
        delta_b1 = [0.0] * self.n_h
        delta_W2 = np.zeros((self.n_o,self.n_h),dtype=np.float32)
        delta_b2 = [0.0] * self.n_o

        #compute desired partial derivates of cost (w.r.t each weight)
        #d_W2 is a (n_o x n_h) matrix
        #d_b2 is a (n_o) vector
        #d_W1 is a (n_h x n_i) matrix
        #d_b1 is a (n_h) vector
        d_W2 = np.zeros((self.n_o,self.n_h),dtype=np.float32)
        d_b2 = [0.0] * self.n_o
        d_W1 = np.zeros((self.n_h,self.n_i),dtype=np.float32)
        d_b1 = [0.0] * self.n_h

        for it in range(0, iter):



            print ("==================Iteration %i==================" % it)

            total_rec_err = 0

            if it == iter -1:
                print "last"

            #for each column (training case) in X
            for idx in range(0, np.shape(self.X)[1]):
                x = self.X[:, idx]

                #perform forward pass
                a2, a3 = self.forward_pass_for_one_case(x)

                err = np.abs(self.Y[idx] - a3)

                #calculate error (delta) for outputs (vector with n_o length)
                delta3 = [0.0] * self.n_o
                for k in range(self.n_o):
                    delta3[k] = (self.Y[idx]-a3[k])*self.dsigmoid(a3[k])

                #calculate the error (delta) for hidden (vector with n_h length)
                delta2 = [0.0] * self.n_h
                for j in range(self.n_h):
                    #accumulate error coming from all nodes in the layer above
                    tmp_error = 0.0
                    for k in range(self.n_o):
                        tmp_error =tmp_error + (self.W2[k][j]*delta3[k])

                    delta2[j] = tmp_error * self.dsigmoid(a2[j])



                for k in range(self.n_o):
                    for j in range(self.n_h):
                        change = delta3[k]*a2[j]
                        self.W2[k][j] = self.W2[k][j] + alpha*change + 0.1*d_W2[k][j]
                        d_W2[k][j] = change
                    changeB = delta3[k]
                    self.b2[k] = self.b2[k] + alpha*changeB + 0.1*d_b2[k]
                    d_b2[k] = changeB


                for j in range(self.n_h):
                    for i in range(self.n_i):
                        change = delta2[j]*x[i,]
                        self.W1[j][i] = self.W1[j][i] + alpha*change + 0.1*d_W1[j][i]
                        d_W1[j][i] = change
                    changeB = delta2[j]
                    self.b1[j] = self.b1[j] + alpha*changeB + 0.1*d_b1[j]
                    d_b1[j] = changeB


                '''
                for k in range(self.n_o):
                    for j in range(self.n_h):
                        delta_W2[k][j] = delta_W2[k][j] + d_W2[k][j]
                        self.W2[k][j] = self.W2[k][j] - alpha*(((1/self.X.shape[1])*delta_W2[k][j]) + w_decay * self.W2[k][j])
                    delta_b2[k] = delta_b2[k] + d_b2[k]
                    self.b2[k] = self.b2[k] - alpha*((1/self.X.shape[1])*delta_b2[k])

                for j in range(self.n_h):
                    for i in range(self.n_i):
                        delta_W1[j][i] = delta_W1[j][i] + d_W1[j][i]
                        self.W1[j][i] = self.W1[j][i] - alpha*(((1/self.X.shape[1])*delta_W1[j][i]) + w_decay * self.W1[j][i])
                    delta_b1[j] = delta_b1[j] + d_b1[j]
                    self.b1[j] = self.b1[j] - alpha*((1/self.X.shape[1])*delta_b1[j])
                '''

                total_rec_err += err

            print ("Total Reconstruction Error: %f" % total_rec_err)

    def test(self):
        for i in range(self.X.shape[1]):
            a2, a3 = self.forward_pass_for_one_case(self.X[:,i])
            print (str(a3[0]))


    def visualize_hidden(self):

        for i in range(0, self.n_h):
            hImg = self.W1[i,:]/LA.norm(self.W1[i,:])
            img = Image.fromarray(np.reshape(hImg, (27, 30))).convert('LA')
            img.save('hImg'+str(i)+'.png')

    #save reconstructed images
    def save_reconstructed(self):


        for i in range(0, 40):
            #hImg = np.zeros((810,), dtype=np.int32)
            a2,a3,a4 = self.forward_pass_for_one_case(self.X[:,i])
            rec_vec = a4*255
            rec_img = np.reshape(rec_vec, (27,30))

            img = Image.fromarray(rec_img).convert('LA')
            img.save('recImg'+str(i+1)+'.png')

#this calls the __init__ method automatically
dA = SimpleAutoEncoder()
dA.load_data()
dA.back_prop()
dA.test()
