import tensorflow as tf

_activation_functions = {
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
}


class Dense:
    def __init__(self, nodes, input_shape, activation_function='relu', name=None):
        self.weights = tf.Variable()
        self.bias = tf.Variable()
        self.activation_function = _activation_functions[activation_function.lower()] if activation_function else None
        self.name = name

    def execute(self, input):
        if self.activation_function:
            return self.activation_function(tf.matmul(input, self.weights) + self.bias, name=self.name)
        else:
            return tf.Variable(tf.matmul(input, self.weights) + self.bias, name=self.name)


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