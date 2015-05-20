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
            shared_x = shared(value=np.asarray(data_x,dtype=config.floatX).T,borrow=True)
            shared_y = shared(value=np.asarray(data_y,dtype=config.floatX).T,borrow=True)

            return shared_x,T.cast(shared_y,'int32')

        train_x,train_y = get_shared_data(train_set)
        valid_x,valid_y = get_shared_data(valid_set)
        test_x,test_y = get_shared_data(test_set)

        all_data = [(train_x,train_y),(valid_x,valid_y),(test_x,test_y)]

        return all_data

    def train_model(self, all_data,iterations=500):

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
            }, mode='DebugMode'
        )

        output_sa1 = train_sa1.outputs

        for epoch in xrange(iterations):
            c=[]
            for batch_index in xrange(n_train_batches):
                c.append(train_sa1(batch_index))

            print 'Training epoch %d, cost ' % epoch, np.mean(c)
        '''
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
        sa2.back_prop(iter=150)
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
        sa3.back_prop(iter=150)
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
        softmax.back_prop_man(iter=500)
        W4_1,b4_1 = softmax.get_params()

        output = np.zeros([self.o_size,data_size],dtype=np.float32)
        idx = 0
        for x in X4.T:
            a2 = softmax.forward_pass(x,W4_1,b4_1)
            output[:,idx] = a2
            idx += 1

        print ("Finished Training")

        outDigits = np.empty((1,self.d_size),dtype=np.int8)
        totCorrect = 0
        for i in range(self.d_size-1):
            maxCol = np.max(output[:,i])
            if maxCol>0.4:
                maxIdx = np.argmax(output[:,i])
                #outDigits[1,i]=maxIdx
                if self.Y_VEC[i]==maxIdx:
                    totCorrect = totCorrect+1
            #else:
                #outDigits[1,i]=-1

        print ("Accuracy: %f" %(1.0*totCorrect/self.d_size))'''

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