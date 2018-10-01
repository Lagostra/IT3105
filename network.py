import tensorflow as tf
import numpy as np
import random
import tflowtools as tft
import matplotlib.pyplot as plt

from layers import Dense

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
    'cross_entropy': lambda y, x: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x)),
}


def load_data(data_source, case_fraction, validation_fraction, test_fraction):
    if type(data_source) == str:
        data = np.loadtxt(data_source, delimiter=',')
    else:
        data = data_source

    data = data[:int(len(data) * case_fraction)]
    train_end = int(len(data) * (1 - max(validation_fraction, 0) - max(test_fraction, 0)))
    validate_end = int(len(data) * (1 - max(test_fraction, 0)))

    train = data[:train_end]
    validate = data if validation_fraction == -1 else data[train_end:validate_end]
    test = data if test_fraction == -1 else data[validate_end:]

    return train, validate, test


def input_target_split(data):
    if len(data[0]) == 2:
        inputs = list(map(lambda x: x[0], data))
        targets = list(map(lambda x: x[1], data))
    else:
        inputs = list(map(lambda x: x[:-1], data))
        targets = list(map(lambda x: x[-1:], data))
    return inputs, targets


def accuracy(targets, predictions):
    n_correct = 0
    for case in zip(targets, predictions):
        correct = True
        for item in zip(case[0], case[1]):
            correct = correct and item[0] == item[1]
        n_correct += int(correct)
    # n_correct = sum(map(lambda x: int(x[0] == x[1]), zip(targets, predictions)))
    return n_correct / len(targets)


class Network:
    inputs = None
    targets = None
    outputs = None
    training_op = None
    loss = None
    accuracy = None
    summaries = []

    def __init__(self, layers, data_source, learning_rate=0.01, steps=1000, minibatch_size=100, optimizer='adam',
                 loss_function='mse', case_fraction=1.0, validation_fraction=0.1, test_fraction=0.2,
                 validation_interval=50, session=None, output_functions=None):

        if type(layers[0]) == int:
            self.layers = []
            for i in range(len(layers) - 1):
                self.layers.append(Dense(layers[i], layers[i + 1]))
        else:
            self.layers = layers

        self.learning_rate = learning_rate
        self.steps = steps
        self.minibatch_size = minibatch_size
        self.optimizer = _optimizers[optimizer]
        self.loss_function = _loss_functions[loss_function]
        self.validation_interval = validation_interval
        self.session = session

        if output_functions == 'argmax_one_hot':
            self.output_functions = [
                lambda x: tf.argmax(x, axis=1),
                lambda x: tf.one_hot(x, self.layers[-1].output_shape[0])
            ]
        else:
            self.output_functions = output_functions

        self.data = load_data(data_source, case_fraction, validation_fraction, test_fraction)

    def build(self):
        self.inputs = tf.placeholder('float', shape=(None,) + self.layers[0].input_shape, name='inputs')
        self.targets = tf.placeholder('float', shape=(None,) + self.layers[-1].output_shape, name='targets')
        x = self.inputs

        for layer in self.layers:
            x = layer.execute(x)

        self.outputs = x
        if self.output_functions:
            for f in self.output_functions:
                self.outputs = f(self.outputs)

        self.accuracy = tf.metrics.accuracy(self.targets, self.outputs)[1]

        self.loss = self.loss_function(self.targets, x)
        self.training_op = self.optimizer(self.learning_rate).minimize(self.loss)

        self.add_summaries()

    def add_summaries(self):
        with tf.name_scope('performance'):
            self.summaries.append(tf.summary.scalar('loss', self.loss))
            self.summaries.append(tf.summary.scalar('accuracy', self.accuracy))

    def predict(self, inputs):
        feed_dict = {
            self.inputs: inputs
        }

        result = self.session.run([self.outputs], feed_dict=feed_dict)[0]
        return result

    def calculate_accuracy(self, data_set):
        inputs, targets = input_target_split(data_set)
        inputs = np.array(inputs)
        predictions = self.predict(inputs)

        return accuracy(targets, predictions)

    def train(self, plot_results=False):
        if not self.session:
            self.session = tft.gen_initialized_session()

        train_set, validate_set, _ = self.data

        errors = []
        validate_accuracies = []

        summaries = tf.summary.merge(self.summaries)

        for i in range(1, self.steps + 1):
            minibatch = random.sample(list(train_set), self.minibatch_size)

            inputs, targets = input_target_split(minibatch)

            feed_dict = {
                self.inputs: inputs,
                self.targets: targets
            }

            _, l, summary = self.session.run([self.training_op, self.loss, summaries], feed_dict=feed_dict)

            self.session.probe_stream.add_summary(summary, global_step=i)

            if i % 50 == 0:
                errors.append(l)
                print('[Step {}] Error: {}'.format(i, l))

            if i % self.validation_interval == 0 and len(validate_set):
                inputs, targets = input_target_split(validate_set)
                feed_dict = {
                    self.inputs: inputs,
                    self.targets: targets
                }
                acc = self.session.run([self.accuracy], feed_dict=feed_dict)
                validate_accuracies.append(acc)

        if plot_results:
            fig, ax1 = plt.subplots()

            ax1.plot(np.arange(0, self.steps, 50), errors, c='blue')
            if len(validate_accuracies):
                ax2 = ax1.twinx()
                ax2.plot(np.arange(0, self.steps, self.validation_interval), validate_accuracies, c='red')
            fig.legend(['Minibatch error', 'Validation accuracy'])
            ax1.set_xlabel('Step')
            plt.show()

        print()

    def test(self):
        _, _, test_set = self.data

        inputs, targets = input_target_split(test_set)

        feed_dict = {
            self.inputs: inputs,
            self.targets: targets
        }

        predictions, error = self.session.run([self.outputs, self.loss], feed_dict=feed_dict)

        try:
            iter(error)
        except TypeError:
            pass
        else:
            error = sum(error)

        print("[Test set] Error: {:.2f}  Accuracy: {:.2f}%".format(error, accuracy(targets, predictions)*100))
        print()
