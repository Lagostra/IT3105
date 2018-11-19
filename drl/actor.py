import numpy as np

from nn.network import Network


class Actor:

    def __init__(self, game, layers=[], checkpoint=None, one_hot_encode_state=True):
        self.game = game
        self.one_hot_encode_state = one_hot_encode_state
        self.layers = layers

        self.network = Network(
            [game.state_size()] + layers + [game.num_possible_moves()],
            [],
            minibatch_size=50,
            steps=1,
            loss_function='cross_entropy',
            validation_fraction=0,
            test_fraction=0,
            learning_rate=0.001,
            optimizer='adam'
        )
        self.network.build()

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def select_move(self, state, stochastic=False):
        possible_moves = self.game.get_moves(state)
        formatted_state = self.game.format_for_nn(state, one_hot_encoded=self.one_hot_encode_state)
        predictions = self.network.predict([formatted_state])[0]

        predictions = predictions[:len(possible_moves)]
        if not stochastic:
            move = np.argmax(predictions)
            return possible_moves[move]

        predictions = np.array(predictions)
        predictions = predictions / predictions.sum()
        move = np.random.choice(np.arange(0, len(predictions)), p=predictions)
        return possible_moves[move]

    def save_checkpoint(self, checkpoint):
        self.network.save(checkpoint)

    def load_checkpoint(self, checkpoint):
        self.network.load(checkpoint)

