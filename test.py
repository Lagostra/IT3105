from network import Network
from layers import Dense
import tflowtools as tft
import numpy as np

layers = [
    Dense(10, 8, name='layer1'),
    Dense(8, 10, name='layer2')
]

from network import input_target_split

network = Network(layers, tft.gen_all_one_hot_cases(10), minibatch_size=10, steps=5000, loss_function='cross_entropy',
                  validation_fraction=0, test_fraction=0)

network.build()
network.train()