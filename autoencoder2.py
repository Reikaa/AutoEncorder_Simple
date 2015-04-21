__author__ = 'Thushan Ganegedara'

import numpy as np
import math
from scipy import misc
from numpy import linalg as LA
from PIL import Image

class SimpleAutoEncoder(object):


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-1.0*x))
        #return math.tanh(x)

    def dsigmoid(self,y):
        return y*(1.0-y)
        #return 1.0-y**2

    #the value specified in the argument for each variable is the default value
    #__init__ is called when the constructor of an object is called (i.e. created an object)
    def __init__(self, n_inputs=810, n_hidden=100, W1=None, W2=None, W3=None, b1=None, b2=None, b3=None):
        self.X = np.zeros((810, 40), dtype=np.float32)

        #define global variables for n_inputs and n_hidden
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs


        #generate random weights for W
        if W1 == None:
            W1 = -0.2 + np.random.random_sample((n_hidden, n_inputs))*0.4
            self.W1 = W1

        if W2 == None:
            W2 = -1.0 + np.random.random_sample((self.n_outputs, n_hidden))*2.0
            self.W2 = W2



        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = -0.05 + np.random.random_sample((n_hidden,)) * 0.1
            self.b1 = b1

        if b2 == None:
            b2 = -1.0 + np.random.random_sample((self.n_outputs,)) * 2.0
            self.b2 = b2




    def load_data(self):

        dir_name = "Data"
        for i in range(1, 41):
            file_name = "\\image_"+str(i)+".png"
            img = misc.imread(dir_name+file_name)
            imgVec = np.reshape(img, (810, 1))
            self.X[:, i-1] = imgVec[:, 0]

        self.X = self.X/255

        return self.X


    def forward_pass_for_one_case(self, x):

        z2 = np.dot(self.W1, x) + self.b1
        a2 = self.sigmoid(z2)

        z3 = np.dot(self.W2, a2)
        a3 = self.sigmoid(z3)

        return a2, a3

    def back_prop(self, X, iter=25, alpha=0.5, w_decay=0.1):


        #gradient descent
        delta_W1 = np.zeros((self.n_hidden, self.n_inputs),dtype=np.float32)
        delta_b1 = np.zeros((self.n_hidden,), dtype=np.float32)
        delta_W2 = np.zeros((self.n_outputs, self.n_hidden),dtype=np.float32)
        #delta_b2 = np.zeros((self.n_hidden2, ),dtype=np.float32)


        for iter in range(0, iter):


            print ("==================Iteration %i==================" % iter)

            total_rec_err = 0

            #for each column (training case) in X
            for idx in range(0, np.shape(X)[1]):
                x = X[:, idx]

                #perform forward pass
                a2, a3 = self.forward_pass_for_one_case(x)

                rec_sqr_err = LA.norm(x - a3)

                #error for each node (delta) in output layer
                delta3 = [0.0] * self.n_outputs
                for k in range(self.n_outputs):
                    delta3[k] = -(x[k] - a3[k]) * self.dsigmoid(a3[k])

                delta2 = [0.0] * self.n_hidden
                for j in range(self.n_hidden):
                    tmp_err = 0.0
                    for k in range(self.n_outputs):
                        tmp_err = tmp_err + (self.W2[k][j]*delta3[k])
                    delta2[j] = tmp_err * self.dsigmoid(a2[j])

                p_deriv_W2 = np.zeros((self.n_outputs,self.n_hidden),dtype=np.float32)
                for k in range(self.n_outputs):
                    for j in range(self.n_hidden):
                        p_deriv_W2[k][j] = delta3[k]*a2[j]
                    #p_deriv_b2 = delta3


                delta_W2 = delta_W2 + p_deriv_W2
                #delta_b2 = delta_b2 + p_deriv_b2

                for k in range(self.n_outputs):
                    for j in range(self.n_hidden):
                        self.W2[k][j] = self.W2[k][j] - alpha*(((1/np.shape(X)[1])*delta_W2[k][j])+(w_decay * self.W2[k][j]))
                #self.b2 = self.b2 - alpha*((1/np.shape(X)[1])*delta_b2)

                p_deriv_W1 = np.zeros((self.n_hidden,self.n_inputs),dtype=np.float32)
                p_deriv_b1 = [0.0] * self.n_hidden
                for j in range(self.n_hidden):
                    for i in range(self.n_inputs):
                        p_deriv_W1[j][i] = delta2[j]*x[i]
                    p_deriv_b1[j] = delta2[j]

                delta_W1 = delta_W1 + p_deriv_W1
                delta_b1 = delta_b1 + p_deriv_b1

                for j in range(self.n_hidden):
                    for i in range(self.n_inputs):
                        self.W1[j][i] = self.W1[j][i] - alpha*(((1/np.shape(X)[1])*delta_W1[j][i])+(w_decay * self.W1[j][i]))
                    self.b1[j] = self.b1[j] - alpha*((1/np.shape(X)[1])*delta_b1[j])

                total_rec_err += rec_sqr_err

                if iter == 20:
                    print "last"

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
            a2,a3 = self.forward_pass_for_one_case(self.X[:,i])
            rec_vec = a3*255
            rec_img = np.reshape(rec_vec, (27,30))

            img = Image.fromarray(rec_img).convert('LA')
            img.save('recImg'+str(i+1)+'.png')

#this calls the __init__ method automatically
dA = SimpleAutoEncoder()
Data = dA.load_data()
dA.back_prop(Data)
#dA.visualize_hidden()
dA.save_reconstructed()