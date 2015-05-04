__author__ = 'Thushan Ganegedara'

import numpy as np
import SparseAutoencoder
from scipy import misc

class StackedAutoencoder(object):

    def __init__(self,data):


    def load_data(self):

        dir_name = "Data"
        for i in range(1, 41):
            file_name = "\\image_"+str(i)+".jpg"
            img = misc.imread(dir_name+file_name)
            imgVec = np.reshape(img, (810, 1))
            self.X[:, i-1] = imgVec[:, 0]

        self.X = self.X/255.0

    def train_model(self):
        in_dim = 27*30
        h1_dim = 18*20
        h2_dim = 9*10

        sa1 = SparseAutoencoder(n_inputs=in_dim,n_hidden=h1_dim,X=self.X)
        sa1.back_prop()
        sa1.forward_pass_for_one_case