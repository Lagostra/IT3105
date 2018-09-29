from network import Network
from layers import Dense, DenseSequence
import tensorflow as tf
import tflowtools as tft
import numpy as np

layers = [
    DenseSequence([10, 8, 10], activation_function='sigmoid')
]

network = Network(layers, tft.gen_all_one_hot_cases(10), minibatch_size=8, steps=2500, loss_function='cross_entropy',
                  validation_fraction=-1, test_fraction=-1, validation_interval=5, output_functions='argmax_one_hot')

network.build()
network.train(plot_results=True)