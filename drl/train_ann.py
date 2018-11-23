import tensorflow as tf

from nn.network import Network
from games.hex import Hex

game = Hex()
state_format = '6-channel'
layers = [1024, 512, 16]


def load_data(path):
    result = []

    with open(path, 'r') as f:
        for line in f:
            state, probs = line.split(';')
            state = list(map(int, state.split(',')))
            player = state[-1]
            board = state[:-1]
            probs = list(map(float, probs.split(',')))
            state = game.format_for_nn((board, player), format=state_format)
            result.append((state, probs))

    return result


if __name__ == '__main__':
    data = load_data('model/100x50-500/replays.txt')

    network = Network(
        [game.state_size(state_format)] + layers + [game.num_possible_moves()],
        data,
        minibatch_size=500,
        steps=5000,
        loss_function='cross_entropy',
        case_fraction=0.1,
        validation_fraction=0,
        validation_interval=1000,
        test_fraction=0,
        learning_rate=0.01,
        optimizer='rmsprop',
        accuracy_argmax=True,
        output_functions=[tf.nn.softmax]
    )

    network.build()
    #network.save('model/pre-trained/step-0')
    # for i in range(4):
    #     network.train(plot_results=True)
    #     network.save(f'model/pre-trained/game_{i*50}')
    network.train(plot_results=True)
    network.save('model/manual/test4')
    #network.test()
