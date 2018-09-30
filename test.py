from network import Network
from layers import Dense, DenseSequence, Dropout
import tensorflow as tf
import tflowtools as tft

layers = [
    DenseSequence([101, 25, 1], activation_function='sigmoid'),
]

network = Network(
    layers,
    tft.gen_symvect_dataset(101, 2000),
    minibatch_size=100,
    steps=2000,
    loss_function='mse',
    test_fraction=0.1,
    validation_fraction=0.1,
    validation_interval=50,
    output_functions=[tf.round]
)

network.build()
network.train(plot_results=True)
network.test()