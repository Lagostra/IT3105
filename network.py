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
    'mse': lambda y, x: tf.reduce_mean(tf.pow(y - x, 2)),
    'mae': tf.losses.absolute_difference,
    'cross_entropy': lambda y, x: tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y),
}


def load_data(data_source, case_fraction, validation_fraction, test_fraction):
    if type(data_source) == str:
        data = np.loadtxt(data_source, delimiter=',')
    else:
        data = np.array(data_source)

    data = data[:int(len(data) * case_fraction)]
    train_end = int(len(data) * (1 - validation_fraction - test_fraction))
    validate_end = int(len(data) * (1 - test_fraction))

    train, validate, test = data[:train_end], data[train_end:validate_end], data[validate_end:]

    return train, validate, test


def input_output_split(data):
    if len(data.shape) == 2:
        return data[:, :-1], data[:, -1:]
    return data[:, 0], data[:, 1]


class Network:
    input = None
    output = None
    target = None
    train_operation = None
    error = None

    def __init__(self, layers, data_source, learning_rate=0.01, optimizer='adam', loss_function='mse',
                 minibatch_size=100, steps=100, case_fraction=1.0, validation_fraction=0.1, test_fraction=0.2,
                 validation_interval=50, session=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = _optimizers[optimizer.lower()](learning_rate=learning_rate)
        self.loss_function = _loss_functions[loss_function.lower()]
        self.minibatch_size = minibatch_size
        self.steps = steps
        self.validation_interval = validation_interval
        self.session = session

        self.data = load_data(data_source, case_fraction, validation_fraction, test_fraction)
        print(self.data)

    def build(self):
        self.input = tf.placeholder('float32', name='input')
        x = self.input

        for layer in self.layers:
            x = layer.execute(x)

        self.output = self.input
        self.target = tf.placeholder('float32', name='target')
        # self.target = tf.squeeze(self.target)
        self.error = tf.reduce_mean(tf.square(self.target - self.output)) # self.loss_function(self.target, self.output)
        self.train_operation = self.optimizer.minimize(self.error)
        test = tf.trainable_variables()

    def train(self):
        if not self.session:
            self.session = tft.gen_initialized_session()

        train_set, validate_set, _ = self.data
        errors = []
        for step in range(self.steps):
            minibatch = np.array(random.sample(train_set, self.minibatch_size))

            inputs, targets = input_output_split(minibatch)

            feed_dict = {self.input: inputs, self.target: targets}

            _, e = self.session.run([self.train_operation, self.error], feed_dict=feed_dict)

            errors.append(e)

            if step % 50 == 0:
                print('Step {}: error = {}'.format(step, e))

        #tft.plot_training_history(errors)

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else tft.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess
