import tensorflow as tf

_activation_functions = {
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
}


class Dense:

    def __init__(self, input_size, output_size, activation_function='relu', name=None, weight_bounds=(0, 1)):
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
        if self.activation_function:
            return self.activation_function(tf.add(tf.matmul(x, self.weights), self.bias), name=self.name)
        return tf.add(tf.matmul(x, self.weights), self.bias, name=self.name)


class DenseSequence:
    def __init__(self, layer_sizes, activation_function='relu', name=None, weight_bounds=(0, 1)):
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
        for layer in self.layers:
            x = layer.execute(x)
        return x


class Dropout:
    def __init__(self, drop_prob=0.2):
        self.keep_prob = 1 - drop_prob

    def execute(self, x):
        return tf.nn.dropout(x, self.keep_prob)
