from network import Network
from layers import Dense, DenseSequence, Dropout
import tensorflow as tf
import tflowtools as tft
from data.mnist.mnist_basics import load_mnist_dataset

layers = [
    # DenseSequence([784, 300, 100, 10], activation_function='relu'),
    Dense(784, 1024, name='layer1'),
    Dense(1024, 64, name='layer2'),
    # Dropout(0.4),
    Dense(64, 10, name='layer2'),
]

network = Network(
    layers,
    load_mnist_dataset(0.1),
    minibatch_size=100,
    steps=1000,
    loss_function='cross_entropy',
    test_fraction=0.1,
    validation_fraction=0.1,
    validation_interval=50,
    output_functions='argmax_one_hot'
)

network.build()
network.train(plot_results=True)
network.test()