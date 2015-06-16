__author__ = 'Thushan Ganegedara'

import numpy as np
import gzip,cPickle
import os
from PIL import Image

class MakeGrayScale(object):
    def load_data(self,dir_name='DataCifar'):

        train_names = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4']
        valid_name = 'data_batch_5'
        test_name = 'test_batch'

        train_data = []
        train_labels = []
        for file_path in train_names:
            f = open(dir_name + os.sep +file_path, 'rb')
            dict = cPickle.load(f)
            train_data.extend(self.turn_gray(dict.get('data')))
            train_labels.extend(dict.get('labels'))
            print "completed train"
        train_set = [train_data,train_labels]

        f = open(dir_name + os.sep +valid_name, 'rb')
        dict = cPickle.load(f)
        valid_set = [self.turn_gray(dict.get('data')),dict.get('labels')]
        print "completed valid"

        f = open(dir_name + os.sep +test_name, 'rb')
        dict = cPickle.load(f)
        test_set = [self.turn_gray(dict.get('data')),dict.get('labels')]
        print "completed test"

        f.close()

        return [train_set,valid_set,test_set]


    def turn_gray(self,data):

        rows = data.shape[0];
        result = np.empty((rows,1024),dtype=np.float16)

        for i in xrange(1024):
            gray = 0.2989 * data[:,i] + 0.5870 * data[:,i+1024] + 0.1140 * data[:,i+2048]
            result[:,i] = gray

        return result

    def save_to_pickle(self,data,f_name):
        with open(f_name, 'wb') as handle:
            cPickle.dump(data, handle)

if __name__ == '__main__':
    mgs = MakeGrayScale()
    tr_set, v_set, te_set = mgs.load_data()
    mgs.save_to_pickle(tr_set,'train_set_bw')
    mgs.save_to_pickle(v_set,'valid_set_bw')
    mgs.save_to_pickle(te_set,'test_set_bw')

