import tensorflow as tf
import numpy as np

_activation_functions = {
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
}


class Dense:
    def __init__(self, input_size, nodes, activation_function='relu', name=None, weight_range=(-1.0, 1.0)):
        weight_name = None if name is None else name + '_weights'
        bias_name = None if name is None else name + '_bias'

        self.weights = tf.Variable(tf.random_uniform([input_size, nodes]))
        self.bias = tf.Variable(tf.random_normal([nodes]))

        # self.weights = tf.Variable(tf.random_uniform([input_size, nodes], minval=weight_range[0], maxval=weight_range[1]), name=weight_name)
        # self.bias = tf.Variable(tf.random_uniform([nodes], minval=weight_range[0], maxval=weight_range[1]), name=bias_name)
        self.activation_function = _activation_functions[activation_function.lower()] if activation_function else None
        self.name = name

    def execute(self, input):
        if self.activation_function:
            return self.activation_function(tf.add(tf.matmul(input, self.weights), self.bias), name=self.name)
        else:
            return tf.add(tf.matmul(input, self.weights), self.bias, name=self.name)


class DenseSequence:
    def __init__(self, layer_sizes, activation_function='relu', name=None):
        self.name = name
        self.layers = []
        for i in range(len(layer_sizes)):
            layer_name = None
            if name:
                layer_name = name + str(i + 1)

            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], activation_function, name=layer_name))

    def execute(self, input):
        for layer in self.layers:
            input = layer.execute(input)
        return input