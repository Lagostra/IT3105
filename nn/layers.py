"""
Contains layer specifications to be used with the network.Network class.
"""
import tensorflow as tf

_activation_functions = {
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
}


class Dense:
    """
    Represents a simple dense neural network layer.
    """

    def __init__(self, input_size, output_size, activation_function='relu', name=None, weight_bounds=(0, 1)):
        """
        Creates a new dense layer.
        :param input_size: The number of inputs fed into the layer.
        :param output_size: The number of nodes in the layer (and the output size from the layer).
        :param activation_function: The activation function applied by the layer, or None for no activation function.
                                    ['relu', 'softmax', 'sigmoid', 'tanh'] Default: 'relu
        :param name: A name for the layer, applied to various tensors. Default: None
        :param weight_bounds: Bounds for the randomly generated weights of the layer. Default: (0, 1)
        """
        self.output = None
        self.input_shape = (input_size,)
        self.output_shape = (output_size,)
        self.name = name
        weight_name = name + '_weights' if name else None
        bias_name = name + '_bias' if name else None

        self.weights = tf.Variable(tf.random_uniform([input_size, output_size],
                                                     minval=weight_bounds[0], maxval=weight_bounds[1]),
                                   name=bias_name,)
        self.bias = tf.Variable(tf.random_uniform([output_size],
                                                  minval=weight_bounds[0], maxval=weight_bounds[1]),
                                name=weight_name)
        self.activation_function = _activation_functions[activation_function] if activation_function else None

    def execute(self, x):
        """
        Adds operations for the layer to the graph.
        :param x: Tensor containing inputs to the layer.
        :return: Tensor containing outputs from the layer.
        """
        if self.activation_function:
            self.output = self.activation_function(tf.add(tf.matmul(x, self.weights), self.bias), name=self.name)
        else:
            self.output = tf.add(tf.matmul(x, self.weights), self.bias, name=self.name)
        return self.output


class DenseSequence:
    """
    Represents a sequence of dense neural network layers with similar parameter settings.
    """

    def __init__(self, layer_sizes, activation_function='relu', name=None, weight_bounds=(0, 1)):
        """
        Creates a sequence of dense layers.
        :param layer_sizes: An array including sizes for all dense layers, starting with initial input size, and ending
                            with final output size.
        :param activation_function: Activation function to be applied in all sublayers.
                                    ['relu', 'softmax', 'sigmoid', 'tanh'] Default: 'relu'
        :param name: A name for the layer, applied to various tensors. Default: None
        :param weight_bounds: Bounds for the randomly generated weights of all sublayers. Default: (0, 1)
        """
        self.output = None
        self.input_shape = (layer_sizes[0],)
        self.output_shape = (layer_sizes[-1],)
        self.name = name
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer_name = None
            if name:
                layer_name = name + str(i + 1)

            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], activation_function,
                                     name=layer_name, weight_bounds=weight_bounds))

    def execute(self, x):
        """
        Adds operations for the layer to the graph.
        :param x: Tensor containing inputs to the layer.
        :return: Tensor containing outputs from the layer.
        """
        for layer in self.layers:
            x = layer.execute(x)
        self.output = x
        return self.output


class Dropout:
    """
    Represents a dropout layer that will apply dropout on a random set of nodes for each step of training,
    automatically scaling output accordingly.
    """

    def __init__(self, drop_prob=0.2):
        """
        Creates a new dropout layer.
        :param drop_prob: The fraction of nodes to be dropped for each step of training.
        """
        self.output = None
        self.keep_prob = 1 - drop_prob

    def execute(self, x):
        """
        Adds operations for the layer to the graph.
        :param x: Tensor containing inputs to the layer.
        :return: Tensor containing outputs from the layer.
        """
        self.output = tf.nn.dropout(x, self.keep_prob)
        return self.output


class Reshape:

    def __init__(self, shape, name=None):
        self.output = None
        self.shape = shape
        self.name = name

    def execute(self, x):
        self.output = tf.reshape(x, self.shape, name=self.name)
        return self.output


class Conv2D:

    def __init__(self, filter, strides=1, padding='SAME', activation_function=None, name=None):
        self.output = None
        self.filter = filter
        self.strides = (1, strides, strides, 1)
        self.padding = padding
        self.activation_function = _activation_functions[activation_function] if activation_function else None
        self.name = name

    def execute(self, x):
        if self.activation_function:
            self.output = self.activation_function(tf.nn.conv2d(x, self.filter, self.strides, self.padding),
                                                   name=self.name)
        else:
            self.output = tf.nn.conv2d(x, self.filter, self.strides, self.padding, name=self.name)
        return self.output


class MaxPool2D:

    def __init__(self, ksize, strides=None, padding='SAME', name=None):
        self.output = None
        strides = strides if strides else ksize
        self.strides = (1, strides, strides, 1)
        self.ksize = (1, ksize, ksize, 1)
        self.padding = padding
        self.name = name

    def execute(self, x):
        self.output = tf.nn.max_pool(x, self.ksize, self.strides, self.padding, name=self.name)
        return self.output


class Flatten:
    def __init__(self, name=None):
        self.output = None

    def execute(self, x):
        self.output = tf.layers.flatten(x, name=self.name)
        return self.output
