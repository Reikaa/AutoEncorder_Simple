__author__ = 'Thushan Ganegedara'

import nlopt
import numpy as np
from DaFacesGPU import SparseAutoencoder
from ReconstructionLayerGPU import ReconstructionLayer
from scipy import optimize
from scipy import misc

import os
from PIL import Image
from numpy import linalg as LA
from math import sqrt,isnan

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


    def __init__(self,in_size=66*37, hidden_size = [500, 500, 250], out_size = 66*37, batch_size = 100, corruption_levels=[0.1, 0.1, 0.1],dropout=True,dropout_rates=[0.1,0.1,0.1]):
        self.i_width = 37
        self.i_height = 66

        self.i_size = in_size
        self.h_sizes = hidden_size
        self.o_size = out_size
        self.batch_size = batch_size
        self.corruption_levels = corruption_levels

        self.n_layers = len(hidden_size)
        self.sa_layers = []
        self.sa_activations_train = []
        self.sa_activations_test = []
        self.thetas = []
        self.thetas_as_blocks = []

        self.dropout = dropout
        self.drop_rates = dropout_rates

        self.cost_fn_names = ['sqr_err', 'neg_log']

        self.x = T.matrix('x')

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

            #if i==0 input is the raw input
            if i==0:
                curr_input_train = self.x
                curr_input_test = self.x
            #otherwise input is the previous layer's hidden activation
            else:
                a2_train = self.sa_layers[-1].get_hidden_act(training=True)
                a2_test = self.sa_layers[-1].get_hidden_act(training=False)
                self.sa_activations_train.append(a2_train)
                self.sa_activations_test.append(a2_test)
                curr_input_train = self.sa_activations_train[-1]
                curr_input_test = self.sa_activations_test[-1]

            sa = SparseAutoencoder(n_inputs=curr_input_size, n_hidden=self.h_sizes[i],
                                   x_train=curr_input_train, x_test=curr_input_test,
                                   dropout=dropout, dropout_rate=self.drop_rates[i])
            self.sa_layers.append(sa)
            self.thetas.extend(self.sa_layers[-1].get_params())
            self.thetas_as_blocks.append(self.sa_layers[-1].get_params())

        #-1 index gives the last element
        a2_train = self.sa_layers[-1].get_hidden_act(training=True)
        a2_test = self.sa_layers[-1].get_hidden_act(training=False)
        self.sa_activations_train.append(a2_train)
        self.sa_activations_test.append(a2_test)

        self.out_sa = ReconstructionLayer(n_inputs=self.h_sizes[-1], n_outputs=self.o_size,
                                          x_train=self.sa_activations_train[-1], x_test = self.sa_activations_test[-1],
                                          dropout=self.dropout,dropout_rate=self.drop_rates[-1])
        self.out_sa_out = self.out_sa.get_hidden_act(training=False)
        self.lam_fine_tune = T.scalar('lam')
        self.fine_cost = self.out_sa.get_finetune_cost(self.x)

        self.thetas.extend(self.out_sa.get_params())

        #measure test performance
        self.error = self.out_sa.get_error(self.x)


    def load_data(self,dir_name='Data'):

        train = np.empty((450,66*37),dtype=np.float32)
        for i in xrange(1, 451):
            file_name = os.sep +"image_"+str(i)+".jpg"
            img = misc.imread(dir_name+file_name)
            imgVec = np.reshape(img, (66*37))/255.0
            train[i-1, :] = imgVec[:]
            #train.append(imgVec)


        valid = np.empty((50,66*37),dtype=np.float32)
        for i in xrange(301, 351):
            file_name = os.sep + "image_"+str(i)+".jpg"
            img = misc.imread(dir_name+file_name)
            imgVec = np.reshape(img, (66*37))/255.0
            valid[i-301, :] = imgVec[:]
            #valid.append(imgVec)


        test = np.empty((100,66*37),dtype=np.float32)
        for i in xrange(351, 451):
            file_name = os.sep + "image_"+str(i)+".jpg"
            img = misc.imread(dir_name+file_name)
            imgVec = np.reshape(img, (66*37))/255.0
            test[i-351, :] = imgVec[:]
            #test.append(imgVec)

        all_data = [train,valid,test]

        def get_shared_data(data_x):
            shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
            return shared_x

        train_x = get_shared_data(train)
        valid_x = get_shared_data(valid)
        test_x = get_shared_data(test)

        all_data = [train_x,valid_x,test_x]

        return all_data

    def greedy_pre_training(self, train_x, batch_size=1, pre_lr=0.25,dropout=True,denoising=False):

        pre_train_fns = []
        index = T.lscalar('index')
        lam = T.scalar('lam')
        beta = T.scalar('beta')
        rho = T.scalar('rho')

        i = 0
        print "\nCompiling functions for DA layers..."
        for sa in self.sa_layers:


            cost, updates = sa.get_cost_and_updates(l_rate=pre_lr, lam=lam, beta=beta, rho=rho, cost_fn=self.cost_fn_names[0], corruption_level=self.corruption_levels[i],denoising=denoising)

            #the givens section in this line set the self.x that we assign as input to the initial
            # curr_input value be a small batch rather than the full batch.
            # however, we don't need to set subsequent inputs to be an only a minibatch
            # because if self.x is only a portion, you're going to get the hidden activations
            # corresponding to that small batch of inputs.
            # Therefore, setting self.x to be a mini-batch is enough to make all the subsequents use
            # hidden activations corresponding to that mini batch of self.x
            sa_fn = function(inputs=[index, Param(lam, default=0.25), Param(beta, default=0.25), Param(rho, default=0.25)], outputs=cost, updates=updates, givens={
                self.x: train_x[index * batch_size: (index+1) * batch_size]
                }
            )

            pre_train_fns.append(sa_fn)
            i = i+1

        return pre_train_fns

    def fine_tuning(self, datasets, batch_size=1, fine_lr=0.2):
        train_set_x = datasets[0]
        valid_set_x = datasets[1]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        gparams = T.grad(self.fine_cost, self.thetas)

        updates = [(param, param - gparam*fine_lr)
                   for param, gparam in zip(self.thetas,gparams)]

        fine_tuen_fn = function(inputs=[index],outputs=self.fine_cost, updates=updates, givens={
            self.x: train_set_x[index * self.batch_size: (index+1) * self.batch_size]
        })

        validation_fn = function(inputs=[index],outputs=self.error, givens={
            self.x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        },name='valid')

        def valid_score():
            return [validation_fn(i) for i in xrange(n_valid_batches)]
        return fine_tuen_fn, valid_score

    def train_model(self, datasets=None, pre_epochs=5, fine_epochs=300, pre_lr=0.4, fine_lr=0.2, batch_size=1, lam=0.0001, beta=0.25, rho = 0.2,dropout=True, denoising=False):

        print "Training Info..."
        print "Batch size: ",
        print batch_size
        print "Pre-training: %f (lr) %i (epochs)" %(pre_lr,pre_epochs)
        print "Fine-tuning: %f (lr) %i (epochs)" %(fine_lr,fine_epochs)
        print "Corruption: ",
        print denoising,
        print self.corruption_levels
        print "Weight decay: ",
        print lam
        print "Dropout: ",
        print dropout
        print "Sparcity: ",
        print "%f (beta) %f (rho)" %(beta,rho)

        train_set_x = datasets[0]
        valid_set_x = datasets[1]
        test_set_x = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        pre_train_fns = self.greedy_pre_training(train_set_x, batch_size=self.batch_size,pre_lr=pre_lr,dropout=dropout,denoising=denoising)

        start_time = time.clock()
        for i in xrange(self.n_layers):

            print "\nPretraining layer %i" %i
            for epoch in xrange(pre_epochs):
                c=[]
                for batch_index in xrange(n_train_batches):
                    c.append(pre_train_fns[i](index=batch_index, lam=lam, beta=beta, rho=rho))

                print 'Training epoch %d, cost ' % epoch,
                print np.mean(c)

            end_time = time.clock()
            training_time = (end_time - start_time)

            print "Training time: %f" %training_time

        max_inp = self.get_input_threshold(datasets[0])
        self.visualize_hidden(max_inp)
        #########################################################################
        #####                          Fine Tuning                          #####
        #########################################################################
        print "\nFine tuning..."

        fine_tune_fn,valid_model = self.fine_tuning(datasets,batch_size=self.batch_size,fine_lr=fine_lr)

        #########################################################################
        #####                         Early-Stopping                        #####
        #########################################################################
        patience = 50 * n_train_batches # look at this many examples
        patience_increase = 2.
        improvement_threshold = 0.0
        #validation frequency - the number of minibatches to go through before checking validation set
        validation_freq = min(n_train_batches*10,patience/2)

        #we want to minimize best_valid_loss, so we shoudl start with largest
        best_valid_loss = np.inf
        test_score = 0.

        done_looping = False
        epoch = 0

        while epoch < fine_epochs and (not done_looping):
            epoch = epoch + 1
            fine_tune_cost = []
            for mini_index in xrange(n_train_batches):
                cost = fine_tune_fn(index=mini_index)
                fine_tune_cost.append(cost)
                #what's the role of iter? iter acts as follows
                #in first epoch, iter for minibatch 'x' is x
                #in second epoch, iter for minibatch 'x' is n_train_batches + x
                #iter is the number of minibatches processed so far...
                iter = (epoch-1) * n_train_batches + mini_index

                # this is an operation done in cycles. 1 cycle is iter+1/validation_freq
                # doing this every epoch
                if (iter+1) % validation_freq == 0:
                    validation_losses = valid_model()
                    curr_valid_loss = np.mean(validation_losses)
                    print 'epoch %i, minibatch %i/%i, validation error is %f' %(epoch, mini_index+1,n_train_batches,curr_valid_loss)

                    if curr_valid_loss < best_valid_loss:

                        if (
                            curr_valid_loss < best_valid_loss * improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_valid_loss = curr_valid_loss
                        best_iter = iter

            print 'Fine tune cost for epoch %i, is %f' %(epoch+1,np.mean(fine_tune_cost))
            #patience is here to check the maximum number of iterations it should check
            #before terminating
            #if patience <= iter:
            #    done_looping = True
            #    break


    def test_model(self,test_set_x,batch_size= 1):

        print '\nTesting the model...'
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar('index')

        #no update parameters, so this just returns the values it calculate
        #without objetvie function minimization
        test_fn = function(inputs=[index], outputs=[self.error], givens={
            self.x: test_set_x[
                index * batch_size: (index + 1) * batch_size
            ]
        }, name='test')

        e=[]
        for batch_index in xrange(n_test_batches):
            err = test_fn(batch_index)
            e.append(err)

        print 'Test Error %f ' % np.mean(e)

    def mkdir_if_not_exist(self, name):
        if not os.path.exists(name):
            os.makedirs(name)

    def visualize_hidden(self,threshold):
        print '\nSaving hidden layer filters...\n'

        #Visualizing 1st hidden layer
        f_name = 'my_filter_layer_0.png'
        im_count = int(sqrt(self.h_sizes[0]))
        image = Image.fromarray(tile_raster_images(
        X=self.sa_layers[0].W1.get_value(borrow=True).T,
        img_shape=(self.i_width, self.i_height), tile_shape=(im_count, im_count),
        tile_spacing=(1, 1)))
        image.save(f_name)

        index = T.lscalar('index')
        max_inputs =[]
        #Higher level hidden layers
        for i in xrange(1,self.n_layers):
            print "Calculating features for higher layers\n"
            inp = np.random.random_sample((self.i_size,))*0.02
            inp = np.asarray(inp,dtype=config.floatX)
            input = shared(value=inp, name='input',borrow=True)

            max_ins = self.nlopt_optimization(input,threshold,i)

            f_name = 'my_filter_layer_'+str(i)+'.png'
            im_count = int(sqrt(self.h_sizes[i]))
            image = Image.fromarray(tile_raster_images(
                X=max_ins,
                img_shape=(self.i_width, self.i_height), tile_shape=(im_count, im_count),
                tile_spacing=(1, 1)))
            image.save(f_name)


    def get_input_threshold(self,train_set_x):
        max_input = np.max(np.sqrt(np.sum(train_set_x.get_value()**2,axis=1)))
        return max_input*0.75


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def cost(self,input,theta_as_blocks,layer_idx,index):

        layer_input = input
        for i in xrange(layer_idx):
            a = self.sigmoid(np.dot(layer_input,theta_as_blocks[i][0]) + theta_as_blocks[i][1])
            layer_input = a

        cost = self.sigmoid(np.dot(layer_input,theta_as_blocks[layer_idx][0]) + theta_as_blocks[layer_idx][1])[index]
        print "         Cost for node %i in layer %i is %f" %(index,layer_idx,cost)
        return -cost

    def cost_prime(self, input, theta_as_blocks, layer_idx, index):

        prime = optimize.approx_fprime(input, self.cost, 0.00000001, theta_as_blocks, layer_idx, index)
        return prime

    def nlopt_optimization(self,input,threshold,layer_idx):

        def nlopt_cost(x,grad,theta_as_blocks,layer_idx,index):

            def get_activation(x, theta_as_blocks, layer_idx, index):
                    layer_input = x
                    for i in xrange(layer_idx):
                        a = self.sigmoid(np.dot(layer_input,theta_as_blocks[i][0]) + theta_as_blocks[i][1])
                        layer_input = a
                    act = self.sigmoid(np.dot(layer_input,theta_as_blocks[layer_idx][0]) + theta_as_blocks[layer_idx][1])[index]
                    return act

            def nlopt_cost_prim(x, theta_as_blocks, layer_idx, index):
                prime = optimize.approx_fprime(x, cost, 0.00000001, theta_as_blocks, layer_idx, index)
                return prime

            if grad.size > 0:
                grad = nlopt_cost_prim(x,theta_as_blocks,layer_idx,index)

            cost = get_activation(x,theta_as_blocks,layer_idx,index)

            print "         nlopt - Cost for node %i in layer %i is %f" %(index,layer_idx,cost)
            return -cost

        #constraint for x
        def con_x_norm(x,grad,threshold):
            if (grad.size>0):
                grad = x/LA.norm(x)
            return LA.norm(x)-threshold

        print 'nlopt - Calculating max activations for layer %i...\n' % layer_idx

        input_arr = input.get_value()
        max_inputs = []

        print 'nlopt - Getting max activations for layer %i\n' % layer_idx
        #creating the ndarray from symbolic theta
        theta_as_blocks_arr = []

        print 'nlopt - Getting theta_as_blocks for layer %i' % layer_idx
        for k in xrange(layer_idx+1):
            print '     nlopt - Getting thetas for layer %i' % k
            theta_as_blocks_arr.append([self.thetas_as_blocks[k][0].get_value(),self.thetas_as_blocks[k][1].get_value()])

        printed_50 = False
        printed_90 = False

        for j in xrange(self.h_sizes[layer_idx]):
            #print '     Getting max input for node %i in layer %i' % (j, layer_idx)
            init_val = input_arr
            opt = nlopt.opt(nlopt.LN_COBYLA, init_val.size)
            opt.set_lower_bounds(np.zeros((init_val.size)))
            opt.set_min_objective(lambda x,grad : nlopt_cost(x,grad,theta_as_blocks_arr,layer_idx,j))
            opt.add_inequality_constraint(lambda x,grad : con_x_norm(x,grad,threshold),1e-8)
            init_val_list = init_val.tolist()
            opt_x = opt.optimize(init_val_list)
            min_x = opt.last_optimum_value()
            max_inputs.append(min_x)

            if j*1.0/self.h_sizes[layer_idx] > .9 and not printed_90:
                print '     90% completed...'
                printed_90 = True
            elif j*1.0/self.h_sizes[layer_idx] > .5 and not printed_50:
                print '     50% completed...'
                printed_50 = True

        return max_inputs

    def get_max_activations(self,input,threshold,layer_idx):

        #constraint for x
        def con_x_norm(x,threshold):
            return threshold-LA.norm(x)
        cons = {'type': 'ineq','fun': con_x_norm,'args': (threshold,)}

        print 'Calculating max activations for layer %i...\n' % layer_idx

        input_arr = input.get_value()
        max_inputs = []

        print 'Getting max activations for layer %i\n' % layer_idx
        #creating the ndarray from symbolic theta
        theta_as_blocks_arr = []

        print 'Getting theta_as_blocks for layer %i' % layer_idx
        for k in xrange(layer_idx+1):
            print '     Getting thetas for layer %i' % k
            theta_as_blocks_arr.append([self.thetas_as_blocks[k][0].get_value(),self.thetas_as_blocks[k][1].get_value()])

        print '\nPerforming optimization (SLSQP) for layer %i...' % layer_idx

        printed_50 = False
        printed_90 = False


        for j in xrange(self.h_sizes[layer_idx]):
            #print '     Getting max input for node %i in layer %i' % (j, layer_idx)
            init_val = input_arr
            res = optimize.minimize(fun=self.cost, x0=init_val, args=(theta_as_blocks_arr,layer_idx,j),
                                    jac=self.cost_prime, method='COBYLA', constraints=cons, options={'maxiter': 5})


            if LA.norm(res.x) > threshold:
                print '     Threshold exceeded node %i layer %i norm %f/%f' % (j, layer_idx, LA.norm(res.x), threshold)
            max_inputs.append(res.x)


            if j*1.0/self.h_sizes[layer_idx] > .9 and not printed_90:
                print '     90% completed...'
                printed_90 = True
            elif j*1.0/self.h_sizes[layer_idx] > .5 and not printed_50:
                print '     50% completed...'
                printed_50 = True

        return np.asarray(max_inputs)

    def save_output_imgs(self,test_x):

        fn = function(inputs=[],outputs=self.out_sa_out,givens={
            self.x: test_x
        })
        out_data_arr = fn()

        self.mkdir_if_not_exist("Reconstructed")
        for i in xrange(out_data_arr.shape[0]):
            img_arr = out_data_arr[i,:]*255.0
            img_mat = np.reshape(img_arr,(self.i_width,self.i_height))
            img = Image.fromarray(img_mat).convert('LA')
            img.save("Reconstructed" + os.sep + 'output_'+str(i)+'.png')

if __name__ == '__main__':
    #sys.argv[1:] is used to drop the first argument of argument list
    #because first argument is always the filename
    try:
        opts,args = getopt.getopt(sys.argv[1:],"h:p:f:b:d:",["w_decay=","early_stopping=","dropout=","corruption=","beta=","rho="])
    except getopt.GetoptError:
        print '<filename>.py -h [<hidden values>] -p <pre-epochs> -f <fine-tuning-epochs> -b <batch_size> -d <data_folder>'
        sys.exit(2)

    #when I run in command line
    if len(opts)!=0:
        lam = 0.0
        dropout = True
        corr_level = [0.1, 0.2, 0.3]
        denoising = False
        beta = 0.0
        rho = 0.0

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
            elif opt == '--w_decay':
                lam = float(arg)
            elif opt == '--dropout':
                dropout_rates = [float(s.strip()) for s in arg.split(',')[1:]]
                if arg=='y':
                    dropout = True
                elif arg == 'n':
                    dropout = False
            elif opt == '--corruption':
                corr_str = arg
                denoise_str = corr_str.split(',')[0]
                if denoise_str=='y':
                    denoising = True
                    corr_level = [float(s.strip()) for s in corr_str.split(',')[1:]]
                else:
                    denoising = False
            elif opt == '--beta':
                beta = float(arg)
            elif opt == '--rho':
                rho = float(arg)

    #when I run in Pycharm
    else:
        lam = 0.0
        hid = [100,100,100]
        pre_ep = 5
        fine_ep = 10
        b_size = 25
        data_dir = 'DataFaces'
        dropout = False
        dropout_rates = [0.2,0.2,0.2,0.2]
        corr_level = [0.1, 0.1, 0.1]
        denoising=False
        beta = 0.2
        rho = 0.2
    sae = StackedAutoencoder(hidden_size=hid, batch_size=b_size, corruption_levels=corr_level,dropout=dropout,dropout_rates=dropout_rates)
    all_data = sae.load_data(data_dir)
    sae.train_model(datasets=all_data, pre_epochs=pre_ep, fine_epochs=fine_ep, batch_size=sae.batch_size, lam=lam, beta=beta, rho=rho, dropout=dropout, denoising=denoising)
    sae.test_model(all_data[2],batch_size=sae.batch_size)

    sae.save_output_imgs(all_data[2])

