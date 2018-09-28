from network import Network
from layers import *
import tflowtools


layers = [
    Dense(8, 10),
    Dense(10, 8)
]

network = Network(layers, tflowtools.gen_all_one_hot_cases(8), steps=1000, minibatch_size=4,
                  loss_function='mse')
network.build()
network.train()