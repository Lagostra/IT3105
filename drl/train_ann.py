from nn.network import Network
from games.hex import Hex

game = Hex()
state_format = '6-channel'
layers = [1000, 500, 1000]


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
        minibatch_size=50,
        steps=1000,
        loss_function='cross_entropy',
        validation_fraction=0,
        test_fraction=0.2,
        learning_rate=0.001,
        optimizer='adam',
        accuracy_argmax=True
    )


    network.build()
    network.train(plot_results=True)
    #network.save('model/manual/test3')
    network.test()
