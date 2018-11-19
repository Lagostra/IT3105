"""
Contains an abstraction of a neural network, allowing the user to quickly create a custom NN architecture,
and run a set of standard operations such as training, evaluation and mapping tests.
"""
import tensorflow as tf
import numpy as np
import random
from nn import tflowtools as tft
import matplotlib.pyplot as plt

from nn.layers import Dense

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


def load_data(data_source, case_fraction, validation_fraction, test_fraction, shuffle=True):
    """
    Loads data from given data source, and splits it into train, validate and test sets.
    :param data_source: The data source, either given as a string to a CSV file, or as an array of cases, each with
    a set of features and a label.
    :param case_fraction: The fraction of the total data that is to be used.
    :param validation_fraction: The fraction of the chosen data to be used for the validation set.
    :param test_fraction: The fraction of the chosen data to be used for the test set.
    :param shuffle: Controls whether the data is shuffled before division or not. Default: True
    :return: A tuple containing the train, validate and test sets.
    """
    if type(data_source) == str:
        data = np.loadtxt(data_source, delimiter=',')
    else:
        data = data_source

    if not len(data):
        return None

    if shuffle:
        random.shuffle(data)

    data = data[:int(len(data) * case_fraction)]
    train_end = int(len(data) * (1 - max(validation_fraction, 0) - max(test_fraction, 0)))
    validate_end = int(len(data) * (1 - max(test_fraction, 0)))

    train = data[:train_end]
    validate = data if validation_fraction == -1 else data[train_end:validate_end]
    test = data if test_fraction == -1 else data[validate_end:]

    return train, validate, test


def input_target_split(data):
    """
    Splits a data set into inputs and targets. Works on both data sets where the cases consists of separate arrays
    of inputs and targets, and data sets where each case is a single array, with the last element being the label.
    :param data: The data set that will be split.
    :return: A tuple containing arrays of the inputs and targets, respectively.
    """
    if len(data[0]) == 2:
        inputs = list(map(lambda x: x[0], data))
        targets = list(map(lambda x: x[1], data))
    else:
        inputs = list(map(lambda x: x[:-1], data))
        targets = list(map(lambda x: x[-1:], data))
    return inputs, targets


def accuracy(targets, predictions):
    """
    Calculates the accuracy of a model, given targets and predictions.
    :param targets: The correct target values.
    :param predictions: The predictions made by a model.
    :return: The accuracy of the model as a decimal number, with 1.0 being perfect accuracy.
    """
    n_correct = 0
    for case in zip(targets, predictions):
        correct = True
        for item in zip(case[0], case[1]):
            correct = correct and item[0] == item[1]
        n_correct += int(correct)
    # n_correct = sum(map(lambda x: int(x[0] == x[1]), zip(targets, predictions)))
    return n_correct / len(targets)


class Network:
    """
    An abstraction of a neural network, allowing the user to quickly create a custom NN architecture,
    and run a set of standard operations such as training, evaluation and mapping tests.
    """

    def __init__(self, layers, data_source, learning_rate=0.01, steps=1000, minibatch_size=100, optimizer='adam',
                 loss_function='mse', case_fraction=1.0, validation_fraction=0.1, test_fraction=0.2,
                 validation_interval=50, session=None, output_functions=None, one_hot_encode_target=False):
        """
        Instantiates a new network.
        :param layers: Specification of the network's layers. Can either be an array of numbers, interpreted as number
        of nodes in a sequence of dense layers, or a sequence of layers as given in the layers module.
        :param data_source: The data source, either as a path to a CSV file, or as a list of cases.
        :param learning_rate: The learning rate used by the neural network optimizer. Default: 0.01
        :param steps: The number of training steps to be carried out. Default: 1000
        :param minibatch_size: The number of cases included in each minibatch during training. Default: 100
        :param optimizer: The optimizer used when training the network ['adam', 'adadelta', 'adagrad',
                            'gradient_descent', 'rmsprop']. Default: 'adam'
        :param loss_function: The loss function used to evaluate the performance of the network
                                ['mse', 'mae', 'cross_entropy']. Default: 'mse'
        :param case_fraction: The fraction of the data set to be used for training, validation and testing sets.
                                Default: 1.0
        :param validation_fraction: The fraction of the data to be used for validation. Default: 0.1
        :param test_fraction: The fraction of the data to be used for testing. Default: 0.2
        :param validation_interval: The interval, in number of minibatches run through the network, between each
                                    validation test. Default: 50
        :param session: A tf.session to be used by the network. If None, a new one is automatically created.
                        Default: None
        :param output_functions: A list of output_functions to be applied as a final stage in the network. Not applied
                                during training. Default: None
        :param one_hot_encode_target: If True, the target will be converted to a one_hot_vector, with length based on
                                        the output size of the network. Requires an integer target. Default: False
        """

        self.inputs = None
        self.targets = None
        self.outputs = None
        self.raw_outputs = None
        self.training_op = None
        self.loss = None
        self.accuracy = None
        self.summaries = []
        self.saver = None
        self.graph = tf.Graph()

        with self.graph.as_default():
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
        self.one_hot_encode_target = one_hot_encode_target

        if output_functions == 'argmax_one_hot':
            self.output_functions = [
                lambda x: tf.argmax(x, axis=1),
                lambda x: tf.one_hot(x, self.layers[-1].output_shape[0])
            ]
        else:
            self.output_functions = output_functions

        self.data = load_data(data_source, case_fraction, validation_fraction, test_fraction)

    def build(self):
        """
        Builds the computation graph of the network. Must be called before training is carried out.
        """
        with self.graph.as_default():
            self.inputs = tf.placeholder('float', shape=(None,) + self.layers[0].input_shape, name='inputs')
            target_shape = (1,) if self.one_hot_encode_target else self.layers[-1].output_shape
            targets = self.targets = tf.placeholder('float', shape=(None,) + target_shape, name='targets')
            if self.one_hot_encode_target:
                targets = tf.squeeze(tf.one_hot(tf.cast(self.targets, 'int32'), self.layers[-1].output_shape[0], axis=1))
            x = self.inputs

            if type(self.layers) == list:
                for layer in self.layers:
                    x = layer.execute(x)
            elif callable(self.layers):
                x = self.layers(x)
            else:
                raise ValueError('layers must be either a list of layers, or a model function')

            self.outputs = self.raw_outputs = x
            if self.output_functions:
                for f in self.output_functions:
                    self.outputs = f(self.outputs)

            self.accuracy = tf.metrics.accuracy(targets, self.outputs, name='accuracy')

            self.loss = self.loss_function(targets, x)
            self.training_op = self.optimizer(self.learning_rate).minimize(self.loss)

            save_variables = []
            for layer in self.layers:
                save_variables.append(layer.weights)
                save_variables.append(layer.bias)
            self.saver = tf.train.Saver(save_variables)

            self.add_summaries()

            if not self.session:
                self.session = tft.gen_initialized_session()

    def load(self, path):
        self.saver.restore(self.session, path)

    def save(self, path):
        self.saver.save(self.session, path)

    def add_summaries(self):
        """
        Adds summaries for loss and accuracy to the graph.
        """
        with self.graph.as_default():
            with tf.name_scope('performance'):
                self.summaries.append(tf.summary.scalar('loss', self.loss))
                self.summaries.append(tf.summary.scalar('accuracy', self.accuracy[0]))

    def predict(self, inputs):
        """
        Makes predictions for the provided inputs using the network.
        :param inputs: Input vectors used as a base for predictions.
        :return: A set of output vectors.
        """
        feed_dict = {
            self.inputs: inputs
        }
        with self.graph.as_default():
            result = self.session.run([self.outputs], feed_dict=feed_dict)[0]
        return result

    def calculate_accuracy(self, data_set):
        """
        Calculates the accuracy of the network over a given data set by creating predictions and comparing to targets.
        :param data_set: The data set to be used for accuracy evaluation.
        :return: Accuracy as a decimal number.
        """
        inputs, targets = input_target_split(data_set)
        inputs = np.array(inputs)
        predictions = self.predict(inputs)

        return accuracy(targets, predictions)

    def train(self, plot_results=False, minibatch=None):
        """
        Trains the network using provided data and hyperparameters.
        :param plot_results: If True, a plot of train set errors and validation accuracy will be created upon finish.
        """
        with self.graph.as_default():
            validate_set = []
            if minibatch:
                train_set = minibatch
            else:
                train_set, validate_set, _ = self.data

            errors = []
            validate_accuracies = []

            summaries = tf.summary.merge(self.summaries)

            for i in range(1, self.steps + 1):
                if not minibatch:
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
                    acc = self.evaluate(validate_set)[1]
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

    def reset_accuracy(self):
        """
        Resets the TensorFlow accuracy counter.
        """
        with self.graph.as_default():
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            self.session.run(running_vars_initializer)

    def evaluate(self, data_set):
        """
        Evaluates the loss and accuracy of a data set by using graph operations.
        :param data_set: The data set that is evaluated.
        :return: A tuple containing the error and accuracy over the data set.
        """
        with self.graph.as_default():
            inputs, targets = input_target_split(data_set)

            feed_dict = {
                self.inputs: inputs,
                self.targets: targets
            }

            self.reset_accuracy()
            _, error = self.session.run([self.accuracy[1], self.loss], feed_dict=feed_dict)
            acc = self.session.run([self.accuracy[0]])[0]
            return error, acc

    def test(self, include_train_set=True):
        """
        Carries out the test stage, evaluating loss and accuracy over the test set and (optionally) the entire
        training set.
        :param include_train_set: If true, testing will also be carried out on the training set as a whole.
        """
        train_set, _, test_set = self.data

        if include_train_set:
            train_error, train_acc = self.evaluate(train_set)

        test_error, test_acc = self.evaluate(test_set)

        if include_train_set:
            print("[Train set] Error: {:.2f}  Accuracy: {:.2f}%".format(train_error, train_acc * 100))

        print("[Test set] Error: {:.2f}  Accuracy: {:.2f}%".format(test_error, test_acc*100))
        print()

    def visualize_weights(self, weight_layers=[], bias_layers=[]):
        """
        Creates visualizations of the weights and biases for the chosen layers. The chosen layers must be dense layers.
        :param weight_layers: A list of indexes to the layers for which weights are to be visualized.
        :param bias_layers: A list of indexes to the layers for which biases are to be visualized.
        """
        with self.graph.as_default():
            for l in weight_layers:
                if l < len(self.layers):
                    layer = self.layers[l]
                    name = layer.name if layer.name else 'Layer ' + str(l)
                    tft.hinton_plot(np.array(self.session.run(layer.weights)), title=name + ' weights')
                    # fig, ax = plt.subplots()
                    # ax.set_title(name + ' weights', pad=30)
                    # im = ax.matshow(self.session.run(layer.weights))
                    # fig.colorbar(im)
                    # plt.show()

            for l in bias_layers:
                if l < len(self.layers):
                    layer = self.layers[l]
                    name = layer.name if layer.name else 'Layer ' + str(l)
                    plt.title(name + ' bias')
                    bias = self.session.run(layer.bias)
                    plt.bar(range(len(bias)), bias)
                    plt.show()

    def mapping_test(self, cases, mapped_layers=[], dendrogram_layers=[]):
        """
        Carries out a mapping tests, creating visualizations of activation levels for chosen layers. Input and output
        activations are always visualized.
        :param cases: Either the number of cases to be mapped given as an integer, or a list of cases to be mapped. If
                        an integer is given, cases will be selected at random from the test set.
        :param mapped_layers: A list of indexes to the layers that activation visualizations should be produced for.
        :param dendrogram_layers: A list of indexes to the layers that dendrograms should be produced for.
        """
        with self.graph.as_default():
            layer_output_tensors = [l.output for l in self.layers]
            layer_names = [l.name if l.name else 'layer' + str(i) for i, l in enumerate(self.layers)]

            if type(cases) == int:
                _, _, test_set = self.data
                cases = random.sample(list(test_set), cases)

            inputs, _ = input_target_split(cases)
            string_inputs = list(map(tft.bits_to_str, inputs))
            feed_dict = {
                self.inputs: inputs
            }

            predictions, *layer_outputs = self.session.run([self.raw_outputs] + layer_output_tensors, feed_dict=feed_dict)

            tft.hinton_plot(np.array(inputs), title='inputs')
            for i in mapped_layers:
                tft.hinton_plot(np.array(layer_outputs[i]), title=layer_names[i])

            for i in dendrogram_layers:
                tft.dendrogram(layer_outputs[i], string_inputs, title=layer_names[i] + ' dendrogram')

            tft.hinton_plot(np.array(predictions), title='outputs')