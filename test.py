from network import Network
from layers import Dense
import tensorflow as tf
import tflowtools as tft
import numpy as np

layers = [
    Dense(10, 8, name='layer1'),
    Dense(8, 10, name='layer2')
]

from network import input_target_split

network = Network(layers, tft.gen_all_one_hot_cases(10), minibatch_size=8, steps=2500, loss_function='cross_entropy',
                  validation_fraction=-1, test_fraction=-1, validation_interval=10,
                  output_functions=[lambda x: tf.argmax(x, axis=1), lambda x: tf.one_hot(x, 10)])

network.build()
network.train(plot_results=True)