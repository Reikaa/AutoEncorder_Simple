__author__ = 'Thushan Ganegedara'

import functools
import itertools

import theano
import theano.tensor as T

import numpy as np

def identity(x):
    return x

def chained_output(layers, x):
    '''
    This method is applying the given transformation (lambda expression) recursively
    to a sequence starting with an initial value (i.e. x)
    :param layers: sequence to perform recursion
    :param x: Initial value to start recursion
    :return: the final value (output after input passing through multiple neural layers)
    '''
    return functools.reduce(lambda acc, layer: layer.output(acc), layers, x)


class Transformer(object):

    #__slots__ save memory by allocating memory only to the varibles defined in the list
    __slots__ = ['layers','_x','_y','_logger']

    def __init__(self,layers):
        self.layers = layers
        self._x = None
        self._y = None
        self._logger = None


    def make_func(self, x, y, batch_size, output, update, transformed_x = identity):
        '''
        returns a Theano function that takes x and y inputs and return the given output using given updates
        :param x: input feature vectors
        :param y: labels of inputs
        :param batch_size: batch size
        :param output: the output to calculate (symbolic)
        :param update: how to get to output from input
        :return: Theano function
        '''
        idx = T.scalar('idx')
        given = {self._x : transformed_x(x[idx * batch_size : (idx + 1) * batch_size]),
                 self._y : y[idx * batch_size : (idx + 1) * batch_size]}

        return theano.function(inputs=[idx],outputs=output, updates=update, givens=given, on_unused_input='warn')

    def process(self, x, y):
        '''
        Visit function with visitor pattern
        :param x:
        :param y:
        :return:
        '''
        pass

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity):
        '''
        Train the network with given params
        :param learning_rate: How fast it learns
        :param x: input feature vectors
        :param y: labels of inputs
        :param batch_size:
        :return: None
        '''
        pass

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        '''
        Validate the network with given parames
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        pass

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        '''
        Calculate error
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        pass


class DeepAutoencoder(Transformer):
    ''' General Deep Autoencoder '''
    def __init__(self,layers, corruption_level, rng):
        super.__init__(layers)
        self._rng = rng
        self._corr_level = corruption_level

        self.theta = None
        self.cost = None
        # Need to find out what cost_vector is used for...
        self.cost_vector = None
        self.validation_error = None

    def process(self, x, y):
        self._x = x
        self._y = y

        # encoding input
        for layer in self.layers:
            W, b_prime = layer.W, layer.b_prime

            #if rng is specified corrupt the inputs
            if self._rng:
                x_tilde = self._rng.binomial(size=(x.shape[0], x.shape[1]), n=1,  p=(1 - self._corruption_level), dtype=theano.config.floatX) * x
                y = layer.output(x_tilde)
            else:
                y = layer.output(x)
                # z = T.nnet.sigmoid(T.dot(y, W.T) + b_prime) (This is required for regularization)

            x = y

        # decoding output and obtaining reconstruction
        for layer in reversed(self.layers):
            W, b_prime = layer.W, layer.b_prime
            x = T.nnet.sigmoid(T.dot(x,W.T) + b_prime)

        # costs
        # cost vector seems to hold the reconstruction error for each training case.
        # this is required for getting inputs with reconstruction error higher than average
        self.cost_vector = T.sum(T.nnet.binary_crossentropy(x, self._x),axis=1)
        self.theta = [ param for layer in self.layers for param in [layer.W, layer.b, layer.b_prime]]
        self.cost = T.mean(self.cost_vector)
        self.validation_error = None
        return None

    def train_func(self, _, learning_rate, x, y, batch_size, transformed_x=identity):
        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]
        return self.make_func(x=x,y=y,batch_size=batch_size,output=None, updates=updates, transformed_x=transformed_x)

    def validate_func(self, _, x, y, batch_size, transformed_x=identity):
        return self.make_func(x=x,y=y,batch_size=batch_size,output=self.validation_error, update=None, transformed_x=transformed_x)

    def get_hard_examples(self, _, x, y, batch_size, transformed_x=identity):
        '''
        Returns the set of training cases (above avg reconstruction error)
        :param _:
        :param x:
        :param y:
        :param batch_size:
        :return:
        '''
        # sort the values by cost and get the top half of it (above average error)
        indexes = T.argsort(self.cost_vector)[(self.cost_vector.shape[0] // 2):]
        return self.make_func(x=x, y=y, batch_size=batch_size, output=[self._x[indexes], self._y[indexes]], update=None, transformed_x=transformed_x)

class StackedAutoencoder(Transformer):
    ''' Stacks a set of autoencoders '''
    def __init__(self, layers, corruption_level, rng):
        super.__init__(layers)
        self._autoencoders = [DeepAutoencoder([layer], corruption_level, rng) for layer in layers]

    def process(self, x, y):
        self._x = x
        self._y = y

        for autoencoder in self._autoencoders:
            autoencoder.process(x,y)

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity):
        return self._autoencoders[arc].train_func(0, learning_rate,x,y,batch_size, lambda x: chained_output(self.layers[:arc],transformed_x(x)))

    def validate_func(self, arc, x, y, batch_size,transformed_x = identity):
        return self._autoencoders[arc].validate_func(0,x,y,batch_size,lambda x: chained_output(self.layers[:arc],transformed_x(x)))