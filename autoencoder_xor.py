__author__ = 'Thushan Ganegedara'

import numpy as np
from scipy import misc
from numpy import linalg as LA
from PIL import Image

class SimpleAutoEncoder(object):


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #the value specified in the argument for each variable is the default value
    #__init__ is called when the constructor of an object is called (i.e. created an object)
    def __init__(self, n_inputs=2, n_hidden=2, W1=None, W2=None, b1=None, b2=None):
        self.X = np.zeros((810, 40), dtype=np.float32)

        #define global variables for n_inputs and n_hidden
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = 1

        #generate random weights for W
        if W1 == None:
            W1 = -0.2 + np.random.random_sample((n_hidden, n_inputs))*(0.2 - (-0.2))
            self.W1 = W1

        if W2 == None:
            W2 = -0.2+ np.random.random_sample((self.n_outputs, n_hidden))*(0.2 - (-0.2))
            self.W2 = W2

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = -0.2 + np.random.random_sample((n_hidden,)) * (0.2 - (-0.2))
            self.b1 = b1

        if b2 == None:
            b2 = -0.2 + np.random.random_sample((self.n_outputs,)) *(0.2 - (-0.2))
            self.b2 = b2



    def load_data(self):

        self.X = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.X = np.transpose(self.X);
        self.Y = np.array([1,0,0,1])


    def forward_pass_for_one_case(self, x=np.zeros((810,), dtype=np.float32)):

        z2 = np.dot(self.W1, x) + self.b1
        a2 = self.sigmoid(z2)

        z3 = np.dot(self.W2, a2) + self.b2
        a3 = self.sigmoid(z3)

        return a2, a3

    def back_prop(self, iter=100, alpha=0.15, w_decay=0.05):

        for i in range(0, iter):

            #gradient descent
            delta_W1 = 0
            delta_b1 = 0
            delta_W2 = 0
            delta_b2 = 0

            print ("==================Iteration %i==================" % i)

            total_rec_err = 0

            if i == iter -1:
                print "last"

            #for each column (training case) in X
            for idx in range(0, np.shape(self.X)[1]):
                x = self.X[:, idx]

                #perform forward pass
                a2, a3 = self.forward_pass_for_one_case(x)

                rec_sqr_err = np.abs(self.Y[idx] - a3)

                #error for each node (delta) in output layer
                delta3 = -(self.Y[idx] - a3) * (a3 * (1-a3))
                delta2 = np.dot(np.transpose(self.W2), delta3) * (a2 * (1-a2))


                p_deriv_W2 = np.dot(delta3[:, None], np.transpose(a2[:, None]))
                p_deriv_b2 = delta3

                delta_W2 = delta_W2 + p_deriv_W2
                delta_b2 = delta_b2 + p_deriv_b2

                self.W2 = self.W2 - alpha*(((1/np.shape(self.X)[1])*delta_W2)+(w_decay * self.W2))
                self.b2 = self.b2 - alpha*((1/np.shape(self.X)[1])*delta_b2)

                p_deriv_W1 = np.dot(delta2[:, None], np.transpose(x[:, None]))
                p_deriv_b1 = delta2

                delta_W1 = delta_W1 + p_deriv_W1
                delta_b1 = delta_b1 + p_deriv_b1

                self.W1 = self.W1 - alpha*(((1/np.shape(self.X)[1])*delta_W1)+(w_decay * self.W1))
                self.b1 = self.b1 - alpha*((1/np.shape(self.X)[1])*delta_b1)

                total_rec_err += rec_sqr_err

            print ("Total Reconstruction Error: %f" % total_rec_err)

    def visualize_hidden(self):

        for i in range(0, self.n_hidden):
            #hImg = np.zeros((810,), dtype=np.int32)
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

