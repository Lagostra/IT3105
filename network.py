import tensorflow as tf
import numpy as np
import random


class Network:
    inputs = None
    targets = None
    outputs = None
    training_op = None
    session = None
    loss = None

    def __init__(self, layers, data, learning_rate=0.01, steps=1000, minibatch_size=100):
        self.layers = layers

        self.data = np.array(data)

        self.learning_rate = learning_rate
        self.steps = steps
        self.minibatch_size = minibatch_size

    def build(self):
        self.inputs = tf.placeholder('float', shape=(None,) + self.layers[0].input_shape)
        self.targets = tf.placeholder('float', shape=(None,) + self.layers[-1].output_shape)
        x = self.inputs

        for layer in self.layers:
            x = layer.execute(x)

        self.outputs = x

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=self.targets)
        self.training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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
