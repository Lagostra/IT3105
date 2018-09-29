from network import Network
from layers import *
import tflowtools as tft


layers = [
    Dense(16, 8, name='layer1', activation_function=None),
    Dense(8, 16, name='layer2')
]

network = Network(layers, tft.gen_all_one_hot_cases(16), steps=50000, minibatch_size=8,
                  loss_function='mse', optimizer='gradient_descent')
network.build()
test = tf.trainable_variables()
network.train()