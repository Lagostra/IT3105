import tensorflow as tf

_activation_functions = {
    'relu': tf.nn.relu
}


class Dense:

    def __init__(self, input_size, output_size, activation_function='relu', name=None):
        self.input_shape = (input_size,)
        self.output_shape = (output_size,)
        self.name = name
        weight_name = name + '_weights' if name else None
        bias_name = name + '_bias' if name else None

        self.weights = tf.Variable(tf.random_uniform([input_size, output_size]), name=bias_name)
        self.bias = tf.Variable(tf.random_uniform([output_size]), name=weight_name)
        self.activation_function = _activation_functions[activation_function] if activation_function else None

    def execute(self, x):
        if self.activation_function:
            return self.activation_function(tf.add(tf.matmul(x, self.weights), self.bias), name=self.name)
        return tf.add(tf.matmul(x, self.weights), self.bias, name=self.name)
