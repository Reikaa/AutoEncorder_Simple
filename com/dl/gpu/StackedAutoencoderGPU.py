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

from theano import function, config, shared, sandbox, Param
import theano.tensor as T
import time

import sys,getopt

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

class StackedAutoencoder(object):


    def __init__(self,in_size=28**2, hidden_size = [500, 500, 250], out_size = 10, batch_size = 100):
        self.i_size = in_size
        self.h_sizes = hidden_size
        self.o_size = out_size
        self.batch_size = batch_size

        self.n_layers = len(hidden_size)
        self.sa_layers = []
        self.sa_activations = []
        self.thetas = []

        self.cost_fn_names = ['sqr_err', 'neg_log']

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.fine_cost = T.dscalar('fine_cost')
        self.error = T.dscalar('test_error')

        print "Network Info:"
        print "Layers: %i" %self.n_layers
        print "Layer sizes: ",
        print self.h_sizes
        print ""
        print "Building the model..."

        for i in xrange(self.n_layers):

            if i==0:
                curr_input_size = self.i_size
            else:
                curr_input_size = self.h_sizes[i-1]

            if i==0:
                curr_input = self.x
            else:
                a2 = self.sa_layers[-1].get_hidden_act()
                self.sa_activations.append(a2)
                curr_input = self.sa_activations[-1]

            sa = SparseAutoencoder(n_inputs=curr_input_size, n_hidden=self.h_sizes[i], input=curr_input)
            self.sa_layers.append(sa)
            self.thetas.extend(self.sa_layers[-1].get_params())

        #-1 index gives the last element
        a2 = self.sa_layers[-1].get_hidden_act()
        self.sa_activations.append(a2)

        self.softmax = SoftmaxClassifier(n_inputs=self.h_sizes[-1], n_outputs=self.o_size, x=self.sa_activations[-1], y=self.y)
        self.lam_fine_tune = T.scalar('lam')
        self.fine_cost = self.softmax.get_cost(self.lam_fine_tune,cost_fn=self.cost_fn_names[1])

        self.thetas.extend(self.softmax.theta)
        self.softmax_out = self.softmax.forward_pass()

        #measure test performance
        self.error = self.softmax.get_error(self.y)

    def load_data(self,file_path='Data\\mnist.pkl.gz'):

        f = gzip.open(file_path, 'rb')
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

    def greedy_pre_training(self, train_x, batch_size=1, pre_lr=0.25):

        pre_train_fns = []
        index = T.lscalar('index')
        lam = T.scalar('lam')

        print "\nCompiling functions for DA layers..."
        for sa in self.sa_layers:


            cost, updates = sa.get_cost_and_updates(l_rate=pre_lr, lam=lam, cost_fn=self.cost_fn_names[0])

            #the givens section in this line set the self.x that we assign as input to the initial
            # curr_input value be a small batch rather than the full batch.
            # however, we don't need to set subsequent inputs to be an only a minibatch
            # because if self.x is only a portion, you're going to get the hidden activations
            # corresponding to that small batch of inputs.
            # Therefore, setting self.x to be a mini-batch is enough to make all the subsequents use
            # hidden activations corresponding to that mini batch of self.x
            sa_fn = function(inputs=[index, Param(lam, default=0.25)], outputs=cost, updates=updates, givens={
                self.x: train_x[index * batch_size: (index+1) * batch_size]
                }
            )

            pre_train_fns.append(sa_fn)

        return pre_train_fns

    def fine_tuning(self, datasets, batch_size=1, learning_rate=0.3):
        (train_set_x, train_set_y) = datasets[0]
        (test_set_x, test_set_y) = datasets[2]
        train_y_mat,valid_y_mat,test_y_mat = datasets[3]

        index = T.lscalar('index')  # index to a [mini]batch

        gparams = T.grad(self.fine_cost, self.thetas)

        updates = [(param, param - gparam*learning_rate)
                   for param, gparam in zip(self.thetas,gparams)]

        fine_tuen_fn = function(inputs=[index, Param(self.lam_fine_tune,default=0.25)],outputs=self.fine_cost, updates=updates, givens={
            self.x: train_set_x[index * self.batch_size: (index+1) * self.batch_size],
            #self.y_mat : train_y_mat[index * self.batch_size: (index+1) * self.batch_size],
            self.y: train_set_y[index * self.batch_size: (index+1) * self.batch_size]
        })

        return fine_tuen_fn

    def train_model(self, datasets=None, pre_epochs=5, fine_epochs=10, pre_lr=0.25, batch_size=1):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        train_y_mat,valid_y_mat,test_y_mat = datasets[3]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        pre_train_fns = self.greedy_pre_training(train_set_x, batch_size=self.batch_size,pre_lr=pre_lr)

        start_time = time.clock()
        for i in xrange(self.n_layers):

            print "\nPretraining layer %i" %i
            for epoch in xrange(pre_epochs):
                c=[]
                for batch_index in xrange(n_train_batches):
                    c.append(pre_train_fns[i](index=batch_index, lam=0.2))

                print 'Training epoch %d, cost ' % epoch,
                print np.mean(c)

            end_time = time.clock()
            training_time = (end_time - start_time)

            print "Training time: %f" %training_time

        print "\nFine tuning..."

        fine_tune_fn = self.fine_tuning(datasets,batch_size=self.batch_size)
        for epoch in xrange(fine_epochs):
            c=[]
            for batch_index in xrange(n_train_batches):
                cost = fine_tune_fn(index=batch_index,lam=0.2)
                c.append(cost)

            print 'Training epoch %d, cost ' % epoch,
            print np.mean(c)


    def test_model(self,test_set_x,test_set_y,batch_size= 1):

        print '\nTesting the model...'
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar('index')

        #no update parameters, so this just returns the values it calculate
        #without objetvie function minimization
        test_fn = function(inputs=[index], outputs=[self.error,self.softmax_out,self.softmax.pred,self.softmax.y], givens={
            self.x: test_set_x[
                index * batch_size: (index + 1) * batch_size
            ],
            self.y: test_set_y[
                index * batch_size: (index + 1) * batch_size
            ]
        }, name='test')

        e=[]
        for batch_index in xrange(n_test_batches):
            err, out, pred, act_y = test_fn(batch_index)
            e.append(err)

        print 'Test Error %f ' % np.mean(e)

    def mkdir_if_not_exist(self, name):
        if not os.path.exists(name):
            os.makedirs(name)

    def visualize_hidden(self):
        print '\nSaving hidden layer filters...'
        i=0
        for sa in self.sa_layers:
            f_name = 'filter_layer_' + str(i) + '.png'
            if i==0:
                im_side = sqrt(self.i_size)
            else:
                im_side = sqrt(self.h_sizes[i-1])
            im_count = int(sqrt(self.h_sizes[i]))
            image = Image.fromarray(tile_raster_images(
            X=sa.W1.get_value(borrow=True).T,
            img_shape=(im_side, im_side), tile_shape=(im_count, im_count),
            tile_spacing=(1, 1)))
            image.save(f_name)
            i += 1


if __name__ == '__main__':
    #sys.argv[1:] is used to drop the first argument of argument list
    #because first argument is always the filename
    try:
        opts,args = getopt.getopt(sys.argv[1:],"h:p:f:b:d:")
    except getopt.GetoptError:
        print '<filename>.py -h [<hidden values>] -p <pre-epochs> -f <fine-tuning-epochs> -b <batch_size> -d <data_folder>'
        sys.exit(2)

    #when I run in command line
    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '-h':
                hid_str = arg
                hid = [int(s.strip()) for s in hid_str.split(',')]
            elif opt == '-p':
                pre_ep = int(arg)
            elif opt == '-f':
                fine_ep = int(arg)
            elif opt == '-b':
                b_size = int(arg)
            elif opt == '-d':
                data_dir = arg
    #when I run in Pycharm
    else:
        hid = [400, 225, 100]
        pre_ep = 5
        fine_ep = 10
        b_size = 50
        data_dir = 'Data\\mnist.pkl.gz'

    sae = StackedAutoencoder(hidden_size=hid, batch_size=b_size)
    all_data = sae.load_data(data_dir)
    sae.train_model(datasets=all_data, pre_epochs=pre_ep, fine_epochs=fine_ep, batch_size=sae.batch_size)
    sae.test_model(all_data[2][0],all_data[2][1],batch_size=sae.batch_size)
    sae.visualize_hidden()