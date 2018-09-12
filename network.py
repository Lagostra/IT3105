import tensorflow as tf

_optimizers = {
    'adam': tf.train.AdamOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'gradient_descent': tf.train.GradientDescentOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
}

_loss_functions = {
    'mse': tf.losses.mean_squared_error,
    'mae': tf.losses.absolute_difference,
    'cross_entropy': tf.losses.softmax_cross_entropy,
}

class Network:

    def __init__(self, layers, learning_rate=0.01, optimizer='adam', loss_function='mse'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = _optimizers[optimizer.lower()](learning_rate=learning_rate)
        self.loss_function = _loss_functions[loss_function.lower()]

    def build(self):
