__author__ = 'Thushan Ganegedara'

import numpy as np
from SparseAutoencoderGPU import SparseAutoencoder
from SoftmaxClassifierGPU import SoftmaxClassifier
from scipy import misc
import os
from PIL import Image
from numpy import linalg as LA
from math import sqrt
import gzip,cPickle
from theano import function, config, shared, sandbox
import theano.tensor as T
import time

class StackedAutoencoder(object):

    def __init__(self):
        self.img_w = 28
        self.img_h = 28
        self.batch_size = 100
        self.d_size = 5000
        self.o_size = 10
        self.W1_1 = None
        self.W2_1 = None
        self.W3_1 = None


    def load_data(self):

        dir_name = "Data"
        f = gzip.open('..\\..\\..\\'+dir_name+'\\mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()


        def get_shared_data(data_xy):
            data_x,data_y = data_xy
            shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
            shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)

            return shared_x,T.cast(shared_y,'int32')

        train_x,train_y = get_shared_data(train_set)
        valid_x,valid_y = get_shared_data(valid_set)
        test_x,test_y = get_shared_data(test_set)

        all_data = [(train_x,train_y),(valid_x,valid_y),(test_x,test_y)]

        return all_data

    def train_model(self, all_data,iterations=50):

        in_dim = self.img_w*self.img_h
        h1_dim = 20**2
        h2_dim = 15**2
        h3_dim = 10**2

        train_x,train_y = all_data[0]
        valid_x,valid_y = all_data[1]
        test_x,test_y = all_data[2]

        n_train_batches = train_x.get_value(borrow=True).shape[0] / self.batch_size

        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()

        sa1 = SparseAutoencoder(n_inputs=in_dim,n_hidden=h1_dim,x=x)

        cost, updates = sa1.get_cost_and_weight_update(l_rate=0.5)

        train_sa1 = function(inputs=[index],outputs=cost,updates=updates,givens={
            x: train_x[index * self.batch_size: (index+1) * self.batch_size]
            }
        )

        start_time = time.clock()
        for epoch in xrange(iterations):
            c=[]
            for batch_index in xrange(n_train_batches):
                c.append(train_sa1(batch_index))

            print 'Training epoch %d, cost ' % epoch, np.mean(c)

        end_time = time.clock()
        training_time = (end_time - start_time)

        print "Training time: %f" %training_time

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
all_data = sae.load_data()
sae.train_model(all_data)
sae.save_hidden(sae.W1_1,"Hidden1")
sae.save_hidden(sae.W2_1,"Hidden2")
#sae.save_hidden(sae.W3_1,"Hidden3")