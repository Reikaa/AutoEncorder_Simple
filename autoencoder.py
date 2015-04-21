__author__ = 'Thushan Ganegedara'

import numpy as np
from scipy import misc
from numpy import linalg as LA
from PIL import Image

class SimpleAutoEncoder(object):


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dsigmoid(self,y):
        return y*(1-y)

    #the value specified in the argument for each variable is the default value
    #__init__ is called when the constructor of an object is called (i.e. created an object)
    def __init__(self, n_inputs=810, n_hidden=200, n_hidden2=100 , W1=None, W2=None, W3=None, b1=None, b2=None, b3=None):
        self.X = np.zeros((810, 40), dtype=np.float32)

        #define global variables for n_inputs and n_hidden
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.n_hidden2 = n_hidden2

        #generate random weights for W
        if W1 == None:
            W1 = -0.2 + np.random.random_sample((n_hidden, n_inputs))*0.4
            self.W1 = W1

        if W2 == None:
            W2 = -1.0+ np.random.random_sample((n_hidden2, n_hidden))*2.0
            self.W2 = W2

        if W3 == None:
            W3 = -2.0 + np.random.random_sample((self.n_outputs, n_hidden2))*4.0
            self.W3 = W3

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = -0.05 + np.random.random_sample((n_hidden,)) * 0.1
            self.b1 = b1

        if b2 == None:
            b2 = -1.0 + np.random.random_sample((self.n_hidden2,)) * 2.0
            self.b2 = b2

        if b3 == None:
            b3 = -2.0 + np.random.random_sample((self.n_outputs,)) * 4.0
            self.b3 = b3



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

        z3 = np.dot(self.W2, a2) + self.b2
        a3 = self.sigmoid(z3)

        z4 = np.dot(self.W3, a3) + self.b3
        a4 = self.sigmoid(z4)

        return a2, a3, a4

    def back_prop(self, X, iter=250, alpha=0.5, w_decay=0.1):



        for i in range(0, iter):

            #gradient descent
            delta_W1 = np.zeros((self.n_hidden, self.n_inputs),dtype=np.float32)
            delta_b1 = np.zeros((self.n_hidden,), dtype=np.float32)
            delta_W2 = np.zeros((self.n_hidden2, self.n_hidden),dtype=np.float32)
            delta_b2 = np.zeros((self.n_hidden2, ),dtype=np.float32)
            delta_W3 = np.zeros((self.n_outputs, self.n_hidden2),dtype=np.float32)
            delta_b3 = np.zeros((self.n_outputs, ),dtype=np.float32)

            print ("==================Iteration %i==================" % i)

            total_rec_err = 0

            #for each column (training case) in X
            for idx in range(0, np.shape(X)[1]):
                x = X[:, idx]

                #perform forward pass
                a2, a3, a4 = self.forward_pass_for_one_case(x)

                rec_sqr_err = LA.norm(x - a4)

                #error for each node (delta) in output layer
                delta4 = -(x - a4) * (a4 * (1.0-a4))
                delta3 = np.dot(np.transpose(self.W3), delta4) * self.dsigmoid(a3)
                delta2 = np.dot(np.transpose(self.W2), delta3) * self.dsigmoid(a2)

                p_deriv_W3 = np.dot(delta4[:, None], np.transpose(a3[:, None]))
                p_deriv_b3 = delta4

                p_deriv_W2 = np.dot(delta3[:, None], np.transpose(a2[:, None]))
                p_deriv_b2 = delta3

                p_deriv_W1 = np.dot(delta2[:, None], np.transpose(x[:, None]))
                p_deriv_b1 = delta2

                delta_W3 = delta_W3 + p_deriv_W3
                delta_b3 = delta_b3 + p_deriv_b3

                delta_W2 = delta_W2 + p_deriv_W2
                delta_b2 = delta_b2 + p_deriv_b2

                delta_W1 = delta_W1 + p_deriv_W1
                delta_b1 = delta_b1 + p_deriv_b1

                self.W3 = self.W3 - alpha*(((1/np.shape(X)[1])*delta_W3)+(w_decay * self.W3))
                self.b3 = self.b3 - alpha*((1/np.shape(X)[1])*delta_b3)

                self.W2 = self.W2 - alpha*(((1/np.shape(X)[1])*delta_W2)+(w_decay * self.W2))
                self.b2 = self.b2 - alpha*((1/np.shape(X)[1])*delta_b2)

                self.W1 = self.W1 - alpha*(((1/np.shape(X)[1])*delta_W1)+(w_decay * self.W1))
                self.b1 = self.b1 - alpha*((1/np.shape(X)[1])*delta_b1)

                total_rec_err += rec_sqr_err

            if i == iter-1:
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
            a2,a3,a4 = self.forward_pass_for_one_case(self.X[:,i])
            rec_vec = a4*255
            rec_img = np.reshape(rec_vec, (27,30))

            img = Image.fromarray(rec_img).convert('LA')
            img.save('recImg'+str(i+1)+'.png')

#this calls the __init__ method automatically
dA = SimpleAutoEncoder()
Data = dA.load_data()
dA.back_prop(Data)
#dA.visualize_hidden()
dA.save_reconstructed()