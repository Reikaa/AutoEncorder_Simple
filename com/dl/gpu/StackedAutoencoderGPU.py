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
        self.y_mat = T.matrix('y_mat')

        self.fine_cost = T.dscalar('fine_cost')

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

        train_y_mat = get_vec_labels(train_set[1])
        valid_y_mat = get_vec_labels(valid_set[1])
        test_y_mat = get_vec_labels(test_set[1])

        all_data = [(train_x,train_y),(valid_x,valid_y),(test_x,test_y),(train_y_mat,valid_y_mat,test_y_mat)]

        return all_data

    def greedy_pre_training(self, train_x , train_y, y_mat_full, batch_size=100):

        pre_train_fns = []

        index = T.lscalar('idx')

        curr_input = self.x

        for i in xrange(self.n_layers):

            print "Compiling function for layer %i..." %i
            if i==0:
                curr_input_size = self.i_size
            else:
                curr_input_size = self.h_sizes[i-1]

            if i > 0:
                a2,a3 = self.sa_layers[-1].forward_pass(curr_input)
                curr_input = a2

            sa = SparseAutoencoder(n_inputs=curr_input_size,n_hidden=self.h_sizes[i],input=curr_input)
            self.sa_layers.append(sa)
            self.thetas.extend(sa.theta)

            cost, updates = sa.get_cost_and_weight_update(l_rate=0.5)

            #the givens section in this line set the self.x that we assign as input to the initial
            # curr_input value be a small batch rather than the full batch.
            # however, we don't need to set subsequent inputs to be an only a minibatch
            # because if self.x is only a portion, you're going to get the hidden activations
            # corresponding to that small batch of inputs.
            # Therefore, setting self.x to be a mini-batch is enough to make all the subsequents use
            # hidden activations corresponding to that mini batch of self.x
            sa_fn = function(inputs=[index], outputs=cost, updates=updates, givens={
                self.x: train_x[index * batch_size: (index+1) * batch_size]
                }
            )

            pre_train_fns.append(sa_fn)

        #-1 index gives the last element
        a2,a3 = self.sa_layers[-1].forward_pass(curr_input)
        curr_input = a2

        softmax = SoftmaxClassifier(n_inputs=self.h_sizes[-1], n_outputs=self.o_size, x=curr_input, y=self.y, y_mat=self.y_mat)
        self.fine_cost = softmax.get_cost(l_rate=0.5)
        self.sa_layers.append(softmax)
        self.thetas.extend(softmax.theta)

        return pre_train_fns

    def fine_tuning(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        train_y_mat,valid_y_mat,test_y_mat = datasets[3]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size


        index = T.lscalar('index')  # index to a [mini]batch

        softmax = self.sa_layers[-1]

        gparams = T.grad(self.fine_cost, self.thetas)

        updates = [(param, param - gparam*learning_rate)
                   for param, gparam in zip(self.thetas,gparams)]

        fine_tuen_fn = function(inputs=[index],outputs=self.fine_cost, updates=updates, givens={
            self.x: train_set_x[index * self.batch_size: (index+1) * self.batch_size],
            self.y: train_set_y[index * self.batch_size: (index+1) * self.batch_size],
            self.y_mat: train_y_mat[index * self.batch_size: (index+1) * self.batch_size,:]
        })

        return fine_tuen_fn

    def train_model(self, datasets=None, pre_epochs=10, pre_lr=0.001, tr_epochs=1000, batch_size=100):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        train_y_mat,valid_y_mat,test_y_mat = datasets[3]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.batch_size

        pre_train_fns = self.greedy_pre_training(train_set_x,train_set_y,train_y_mat)

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

        print "Fine tuning..."

        fine_tune_fn = self.fine_tuning(datasets,100,0.5)
        for epoch in range(pre_epochs):
            c=[]
            for batch_index in xrange(n_train_batches):
                c.append(fine_tune_fn(batch_index))

            print 'Training epoch %d, cost ' % epoch, np.mean(c)
    def mkdir_if_not_exist(self, name):
        if not os.path.exists(name):
            os.makedirs(name)


sae = StackedAutoencoder()
all_data = sae.load_data()
sae.train_model(datasets=all_data)

#sae.save_hidden(sae.W3_1,"Hidden3")