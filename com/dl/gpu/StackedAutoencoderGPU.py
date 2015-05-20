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

    def __init__(self,in_size=784, hidden_size = [500, 250, 125], out_size = 10, batch_size = 100):
        self.i_size = in_size
        self.h_sizes = hidden_size
        self.o_size = out_size
        self.batch_size = batch_size

        self.n_layers = len(hidden_size)
        self.sa_layers = []
        self.sa_activations = []
        self.thetas = []

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

    def greedy_pre_training(self, train_x ,iterations=50):

        pre_train_fns = []

        x = T.matrix('x')
        y = T.ivector('y')

        curr_input = self.train_x

        for i in xrange(self.n_layers):

            if i==0:
                curr_input_size = self.i_size
            else:
                curr_input_size = self.h_sizes[i-1]

            if i > 0:
                a2,a3 = self.sa_layers[i-1].forward_pass(curr_input)
                curr_input = a2

            index = T.lscalar()

            sa = SparseAutoencoder(n_inputs=curr_input_size,n_hidden=self.h_sizes[i],x=curr_input)
            self.sa_layers.append(sa)

            cost, updates = sa.get_cost_and_weight_update(l_rate=0.5)

            sa_fn = function(inputs=[index],outputs=cost,updates=updates,givens={
                x: curr_input[index * self.batch_size: (index+1) * self.batch_size]
                }
            )

            pre_train_fns.append(sa_fn)

        return pre_train_fns

    def test_model(self, pre_epochs=50, pre_lr=0.001, tr_epochs=1000, batch_size=100):
        datasets = self.load_data()

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.batch_size

        pre_train_fns = self.greedy_pre_training(train_set_x)

        start_time = time.clock()
        for i in xrange(self.da_layers):

            for epoch in xrange(pre_epochs):
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