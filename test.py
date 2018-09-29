from network import Network
from layers import Dense
import tflowtools as tft

layers = [
    Dense(16, 8, name='layer1'),
    Dense(8, 16, name='layer2')
]

network = Network(layers, tft.gen_all_one_hot_cases(16), minibatch_size=8, steps=5000)

network.build()
network.train()