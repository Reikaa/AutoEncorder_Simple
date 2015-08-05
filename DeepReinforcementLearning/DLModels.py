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

def iterations_shim(func, iterations):
    '''
    Repeated calls to the same function
    :param func: The function
    :param iterations: number of times to call the function
    :return:
    '''

    def function(i):
        for _ in range(iterations):
            func(i)
    return func


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

    def validate_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._autoencoders[arc].validate_func(0,x,y,batch_size,lambda x: chained_output(self.layers[:arc],transformed_x(x)))

class Softmax(Transformer):

    def __init__(self, layers, iterations):
        super.__init__(layers)

        self.theta = None
        self._errors = None
        self.cost_vector = None
        self.cost = None
        self.iterations = iterations

    def process(self, x, y):
        self._x = x
        self._y = y

        p_y_given_x = T.nnet.softmax(chained_output(self.layers, x))

        results = T.argmax(p_y_given_x, axis=1)

        self.theta = [param for layer in self.layers for param in [layer.W, layer.b]]
        self.errors = T.mean(T.neq(results,y))
        self.cost_vector = -T.log(p_y_given_x)[T.arrange(y.shape[0]), y]
        self.cost = T.mean(self.cost_vector)

        return None

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity, iterations=None):

        if iterations is None:
            iterations = self.iterations

        updates = [(param, param - learning_rate*grad) for param, grad in zip(self.theta, T.grad(self.cost,wrt=self.theta))]

        train = self.make_func(x,y,batch_size,None,updates,transformed_x)
        return iterations_shim(train, iterations)

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self.make_func(x,y,batch_size,self.cost,None,transformed_x)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self.make_func(x,y,batch_size,self._errors,None, transformed_x)

class Pool(object):
    ''' A ring buffer (Acts as a Queue) '''
    __slots__ = ['size', 'max_size', 'position', 'data', 'data_y', '_update']

    def __init__(self, row_size, max_size):
        self.size = 0
        self.max_size = max_size
        self.position = 0

        self.data = theano.shared(np.empty(max_size, row_size, dtype=theano.config.floatX), 'pool' )
        self.data_y = theano.shared(np.empty(max_size, dtype='int32'), 'pool_y')

        x = T.matrix('new_data')
        y = T.ivector('new_data_y')
        pos = T.iscalar('update_index')

        update = [(self.data, T.set_subtensor(self.data[pos:pos+x.shape[0]],x)),
            (self.data_y, T.set_subtensor(self.data_y[pos:pos+y.shape[0]],y))]

        self._update = theano.function([pos, x, y], updates=update)

    def add(self, x, y, rows=None):

        if not rows:
            rows = x.shape[0]

        if rows > self.max_size:
            x = x[rows - self.max_size]
            y = y[rows - self.max_size]

        if rows+ self.position > self.max_size:
            available_size = self.max_size - self.position
            self._ring_add(x[:available_size], y[:available_size])
            x = x[available_size:]
            y = y[available_size:]

        self._ring_add(x,y)

    def clear(self):
        self.size = 0
        self.position = 0

    def _ring_add(self, x, y):
        self._update(self.position, x, y)
        self.size = min(self.size + x.shape[0], self.max_size)
        self.position = (self.position + x.shape[0]) % self.max_size

class MergeIncrementingAutoencoder(Transformer):

    __slots__ = ['_autoencoder', '_layered_autoencoders', '_combined_objective', '_softmax', 'lam', '_updates', '_givens', 'rng', 'iterations']

    def __init__(self, layers, corruption_level, rng, lam, iterations):
        super.__init__(layers)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._layered_autoencoders = [DeepAutoencoder([self.layers[i]], corruption_level, rng)
                                       for i, layer in enumerate(self.layers[:-1])]
        self._softmax = Softmax(layers)
        self._combined_objective = CombinedObjective(layers, corruption_level, rng, lam, iterations)
        self.lam = lam
        self.iterations = iterations
        self.rng = np.random.RandomState(0)

    def process(self, x, y):
        self._x = x
        self._y = y
        self._autoencoder.process(x,y)
        self._softmax.process(x,y)
        self._combined_objective.process(x,y)
        for ae in self._layered_autoencoders:
            ae.process(x,y)

    def merge_inc_func(self, learning_rate, batch_size, x, y):

        m = T.matrix('m')
        # map operation applies a certain function to a sequence. This is the upper part of cosine dist eqn
        m_dists, _ = theano.map(lambda v: T.sqrt(T.dot(v, v.T)), m)
        # dimshuffle(0,'x') is converting N -> Nx1
        m_cosine = (T.dot(m, m.T)/m_dists) / m_dists.dimshuffle(0,'x')
        m_ranks = T.argsort((m_cosine - T.tri(m.shape[0]) * np.finfo(theano.config.floatX).max).flatten())[(m.shape[0] * (m.shape[0]+1)) // 2:]

class CombinedObjective(Transformer):

    def __init__(self, layers, corruption_level, rng, lam, iterations):
        super.__init__(layers)

        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._softmax = Softmax(layers)
        self.lam = lam
        self.iterations = iterations

    def process(self, x, yy):
        self._x = x
        self._y = yy

        self._autoencoder.process(x,yy)
        self._softmax.process(x,yy)

    def train_func(self, arc, learning_rate, x, y, batch_size, transformed_x=identity, iterations = None):

        if iterations is None:
            iterations = self.iterations

        combined_cost = self._softmax.cost + self.lam * self._autoencoder.cost

        theta = []
        for layer in self.layers[:-1]:
            theta += [layer.W, layer.b, layer.b_prime]
        theta += [self.layers[-1].W, self.layers[-1].b] #softmax layer

        update = [(param, param - learning_rate * grad) for param, grad in zip(theta, T.grad(combined_cost,wrt=theta))]
        func = self.make_func(x, y, batch_size, None, update, transformed_x)
        return iterations_shim(func, iterations)

    def validate_func(self, arc, x, y, batch_size, transformed_x=identity):
        return self._softmax.validate_func(arc, x, y, batch_size, transformed_x)

    def error_func(self, arc, x, y, batch_size, transformed_x = identity):
        return self._softmax.error_func(arc, x, y, batch_size, transformed_x)


class DeepReinforcementLearningModel(Transformer):

    def __init__(self, layers, corruption_level, rng, iterations, lam, mi_batch_size, pool_size, controller):

        super.__init__(layers)

        self._mi_batch_size = mi_batch_size
        self._controller = controller
        self._autoencoder = DeepAutoencoder(layers[:-1], corruption_level, rng)
        self._softmax = CombinedObjective(layers, corruption_level, rng, lam, iterations)
        self._merge_inc =
