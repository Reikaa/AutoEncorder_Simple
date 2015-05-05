__author__ = 'Thushan Ganegedara'

import numpy as np
from SparseAutoencoder import SparseAutoencoder
from SoftmaxClassifier import SoftmaxClassifier
from scipy import misc
import os
from PIL import Image
from numpy import linalg as LA
from math import sqrt
import gzip,cPickle

class StackedAutoencoder(object):

    def __init__(self):
        self.img_w = 28
        self.img_h = 28
        self.d_size = 5000
        self.o_size = 10
        self.X = np.zeros((self.img_w*self.img_h, self.d_size),dtype=np.float32)
        self.Y = np.zeros((self.o_size,self.d_size),dtype=np.float32)
        self.Y_VEC = np.zeros((self.d_size,),dtype=np.float32)
        self.W1_1 = None
        self.W2_1 = None
        self.W3_1 = None


    def load_data(self):

        dir_name = "Data"
        f = gzip.open(dir_name+'\\mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X = np.transpose(train_set[0][1:self.d_size,:])
        self.Y_VEC = train_set[1][1:self.d_size]

        idx = 0
        for y in train_set[1][1:self.d_size]:
            tmp_arr = [0.0] * self.o_size
            tmp_arr[y] = 1.0
            self.Y[:,idx] = tmp_arr
            idx += 1

        print ("Last")



    def train_model(self):
        in_dim = self.img_w*self.img_h
        h1_dim = 15*15
        h2_dim = 10*10
        h3_dim = 8*8


        data_size = self.X.shape[1]

        sa1 = SparseAutoencoder(n_inputs=in_dim,n_hidden=h1_dim,X=self.X)
        sa1.back_prop(iter=3)
        W1_1,b1_1,W1_2,b1_2 = sa1.get_params()
        self.W1_1 = W1_1

        print ("Trained 1st AE...")
        X2 = np.zeros([h1_dim,data_size],dtype=np.float32)
        idx = 0
        for x in self.X.T:
            a2,a3 = sa1.forward_pass_for_one_case(x,W1_1,b1_1,W1_2,b1_2)
            X2[:,idx] = a2
            idx += 1

        print ("Inputs for 2nd AE created. Size (%i, %i)" %(X2.shape[0],X2.shape[1]))
        sa2 = SparseAutoencoder(n_inputs=h1_dim,n_hidden=h2_dim,X=X2)
        sa2.back_prop(iter=3)
        W2_1,b2_1,W2_2,b2_2 = sa2.get_params()
        self.W2_1 = W2_1

        print ("Trained 2nd AE...")

        X3 = np.zeros([h2_dim,data_size],dtype=np.float32)
        idx = 0
        for x in X2.T:
            a2,a3 = sa2.forward_pass_for_one_case(x,W2_1,b2_1,W2_2,b2_2)
            X3[:,idx] = a2
            idx += 1

        print ("Inputs for 2nd AE created. Size (%i, %i)" %(X3.shape[0],X3.shape[1]))
        sa3 = SparseAutoencoder(n_inputs=h2_dim, n_hidden=h3_dim, X=X3)
        sa3.back_prop(iter=3)
        W3_1,b3_1,W3_2,b3_2 = sa3.get_params()
        self.W3_1 = W3_1

        print ("Trained 3rd AE...")

        X4 = np.zeros([h3_dim,data_size],dtype=np.float32)
        idx = 0
        for x in X3.T:
            a2,a3 = sa3.forward_pass_for_one_case(x,W3_1,b3_1,W3_2,b3_2)
            X4[:,idx] = a2
            idx += 1

        print ("Inputs for 3rd AE created. Size (%i, %i)" %(X4.shape[0],X4.shape[1]))

        softmax = SoftmaxClassifier(n_inputs=h3_dim, n_outputs=self.o_size, X=X4, Y=self.Y_VEC)
        softmax.back_prop(iter=3)

        print ("Finished Training")

    def mkdir_if_not_exist(self, name):
        if not os.path.exists(name):
            os.makedirs(name)

    def save_hidden(self, W, dir="Hidden"):
        h_dir = dir
        width = int(sqrt(W.shape[1]))
        self.mkdir_if_not_exist(h_dir)
        for i in range(W.shape[0]):
            hImg = (W[i,:]/LA.norm(W[i, :]))*500.0
            img = Image.fromarray(np.reshape(hImg, (width, width))).convert('LA')
            img.save(h_dir + "\\" + 'hImg'+str(i)+'.png')

sae = StackedAutoencoder()
sae.load_data()
sae.train_model()
sae.save_hidden(sae.W1_1,"Hidden1")
sae.save_hidden(sae.W2_1,"Hidden2")
#sae.save_hidden(sae.W3_1,"Hidden3")