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

    def __init__(self,in_size=784, hidden_size = [250, 250, 125], out_size = 10, batch_size = 100):
        self.i_size = in_size
        self.h_sizes = hidden_size
        self.o_size = out_size
        self.batch_size = batch_size

        self.n_layers = len(hidden_size)
        self.sa_layers = []
        self.sa_activations = []
        self.thetas = []

        self.x = T.matrix('x')
        self.y = T.ivector('y')

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

        def get_vec_labels(y):
            y_mat = []
            for i in y:
                y_i = y[i]
                y_vec  = [0.0] * self.o_size
                y_vec[y_i] = 1.0
                y_mat.append(y_vec)

            return shared(value=np.asarray(y_mat, dtype=config.floatX), name = 'y_mat', borrow=True)

        train_x,train_y = get_shared_data(train_set)
        valid_x,valid_y = get_shared_data(valid_set)
        test_x,test_y = get_shared_data(test_set)

        tmp_train_y = train_set[1]
        y_mat = get_vec_labels(tmp_train_y)

        all_data = [(train_x,train_y),(valid_x,valid_y),(test_x,test_y),y_mat]

        return all_data

    def greedy_pre_training(self, train_x , train_y, y_mat_full, batch_size=100):

        pre_train_fns = []

        y_mat = T.matrix('y_mat')
        idx = T.lscalar('idx')

        curr_input = self.x

        for i in xrange(self.n_layers):

            print "Compiling function for layer %i..." %i
            if i==0:
                curr_input_size = self.i_size
            else:
                curr_input_size = self.h_sizes[i-1]

            if i > 0:
                a2,a3 = self.sa_layers[i-1].forward_pass(curr_input)
                curr_input = a2

            sa = SparseAutoencoder(n_inputs=curr_input_size,n_hidden=self.h_sizes[i],input=curr_input)
            self.sa_layers.append(sa)

            cost, updates = sa.get_cost_and_weight_update(l_rate=0.5)

            sa_fn = function(inputs=[idx], outputs=cost, updates=updates, givens={
                self.x: curr_input[idx * batch_size: (idx+1) * batch_size]
                }
            )

            pre_train_fns.append(sa_fn)

        #-1 index gives the last element
        a2,a3 = self.sa_layers[-1].forward_pass(curr_input)
        curr_input = a2

        softmax = SoftmaxClassifier(n_inputs=self.h_sizes[-1], n_outputs=self.o_size, x=curr_input, y=train_y, y_mat=y_mat_full)
        cost, updates = softmax.get_cost_and_weight_update(l_rate=0.5)

        soft_fn = function(inputs=[idx],outputs=cost,updates=updates, givens={
            curr_input: curr_input[idx * self.batch_size: (idx+1) * self.batch_size],
            y_mat: y_mat_full[idx * self.batch_size: (idx+1) * self.batch_size,:]
        })

        pre_train_fns.append(soft_fn)

        return pre_train_fns

    def train_model(self, datasets=None, pre_epochs=50, pre_lr=0.001, tr_epochs=1000, batch_size=100):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        y_mat = datasets[3]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.batch_size

        pre_train_fns = self.greedy_pre_training(train_set_x,train_set_y,y_mat)

        start_time = time.clock()
        for i in xrange(self.n_layers):

            print "Pretraining layer %i" %i
            for epoch in range(pre_epochs):
                c=[]
                for batch_index in xrange(n_train_batches):
                    c.append(pre_train_fns[i](batch_index))

                print 'Training epoch %d, cost ' % epoch, np.mean(c)

            end_time = time.clock()
            training_time = (end_time - start_time)

            print "Training time: %f" %training_time

    def mkdir_if_not_exist(self, name):
        if not os.path.exists(name):
            os.makedirs(name)


sae = StackedAutoencoder()
all_data = sae.load_data()
sae.train_model(datasets=all_data)

#sae.save_hidden(sae.W3_1,"Hidden3")