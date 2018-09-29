import tensorflow as tf
import numpy as np
import random


_optimizers = {
    'adam': tf.train.AdamOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'gradient_descent': tf.train.GradientDescentOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
}

_loss_functions = {
    'mse': lambda y, x: tf.reduce_mean(tf.square(y - x)),
    'mae': tf.losses.absolute_difference,
    'cross_entropy': lambda y, x: tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x),
}

class Network:
    inputs = None
    targets = None
    outputs = None
    training_op = None
    session = None
    loss = None

    def __init__(self, layers, data, learning_rate=0.01, steps=1000, minibatch_size=100, optimizer='adam',
                 loss_function='mse'):
        self.layers = layers
        self.data = np.array(data)

        self.learning_rate = learning_rate
        self.steps = steps
        self.minibatch_size = minibatch_size
        self.optimizer = _optimizers[optimizer]
        self.loss_function = _loss_functions[loss_function]

    def build(self):
        self.inputs = tf.placeholder('float', shape=(None,) + self.layers[0].input_shape)
        self.targets = tf.placeholder('float', shape=(None,) + self.layers[-1].output_shape)
        x = self.inputs

        for layer in self.layers:
            x = layer.execute(x)

        self.outputs = x

        self.loss = self.loss_function(self.targets, x)
        self.training_op = self.optimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(1, self.steps + 1):
                minibatch = np.array(random.sample(list(self.data), self.minibatch_size))

                feed_dict = {
                    self.inputs: minibatch[:, 0],
                    self.targets: minibatch[:, 1]
                }

                _, l = sess.run([self.training_op, self.loss], feed_dict=feed_dict)

                if i % 50 == 0:
                    print('[Step {}] Loss: {}'.format(i, l))

            softmax = tf.nn.softmax(self.outputs)
            argmax = tf.argmax(softmax)
            one_hot = tf.one_hot(argmax, 8)
            result = sess.run([argmax], feed_dict={self.inputs: self.data[:, 0]})
            print(result)
            print(self.inputs)
