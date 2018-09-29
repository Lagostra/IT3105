import tensorflow as tf
import numpy as np
import random
import tflowtools as tft


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


def load_data(data_source, case_fraction, validation_fraction, test_fraction):
    if type(data_source) == str:
        data = np.loadtxt(data_source, delimiter=',')
    else:
        data = data_source

    data = data[:int(len(data) * case_fraction)]
    train_end = int(len(data) * (1 - validation_fraction - test_fraction))
    validate_end = int(len(data) * (1 - test_fraction))

    train, validate, test = data[:train_end], data[train_end:validate_end], data[validate_end:]

    return train, validate, test


def input_target_split(data):
    if len(data[0]) == 2:
        inputs = list(map(lambda x: x[0], data))
        targets = list(map(lambda x: x[1], data))
    else:
        inputs = list(map(lambda x: x[:-1], data))
        targets = list(map(lambda x: x[-1:], data))
    return inputs, targets


class Network:
    inputs = None
    targets = None
    outputs = None
    training_op = None
    loss = None

    def __init__(self, layers, data_source, learning_rate=0.01, steps=1000, minibatch_size=100, optimizer='adam',
                 loss_function='mse', case_fraction=1.0, validation_fraction=0.1, test_fraction=0.2,
                 validation_interval=50, session=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.steps = steps
        self.minibatch_size = minibatch_size
        self.optimizer = _optimizers[optimizer]
        self.loss_function = _loss_functions[loss_function]
        self.validation_interval = validation_interval
        self.session = session

        self.data = load_data(data_source, case_fraction, validation_fraction, test_fraction)

    def build(self):
        self.inputs = tf.placeholder('float', shape=(None,) + self.layers[0].input_shape, name='inputs')
        self.targets = tf.placeholder('float', shape=(None,) + self.layers[-1].output_shape, name='targets')
        x = self.inputs

        for layer in self.layers:
            x = layer.execute(x)

        self.outputs = x

        self.loss = self.loss_function(self.targets, x)
        self.training_op = self.optimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        if not self.session:
            self.session = tft.gen_initialized_session()

        train_set, validate_set, _ = self.data

        for i in range(1, self.steps + 1):
            minibatch = random.sample(list(train_set), self.minibatch_size)

            inputs, targets = input_target_split(minibatch)

            feed_dict = {
                self.inputs: inputs,
                self.targets: targets
            }

            _, l = self.session.run([self.training_op, self.loss], feed_dict=feed_dict)

            if i % 50 == 0:
                print('[Step {}] Loss: {}'.format(i, l))

        softmax = tf.nn.softmax(self.outputs)
        argmax = tf.argmax(softmax)
        one_hot = tf.one_hot(argmax, 8)
        result = self.session.run([argmax], feed_dict={self.inputs: input_target_split(train_set)[0]})
        print(result)
        print(self.inputs)
