from collections import deque
import numpy as np
import random

from nn.network import Network


class Actor:

    def __init__(self, game, layers=[], minibatch_size=50):
        self.game = game
        self.replay_buffer = deque(maxlen=20000)
        self.minibatch_size = minibatch_size

        self.network = Network(
            [game.state_size()] + layers + [game.num_possible_moves()],
            self.replay_buffer,
            minibatch_size=50,
            steps=1,
            loss_function='cross_entropy',
            validation_fraction=0,
            test_fraction=0,
            learning_rate=0.001,
            optimizer='adam'
        )
        self.network.build()

    def select_move(self, state, stochastic=False):
        possible_moves = self.game.get_moves(state)
        formatted_state = self.game.format_for_nn(state)
        predictions = self.network.predict([formatted_state])[0]

        predictions = predictions[:len(possible_moves)]
        if not stochastic:
            move = np.argmax(predictions)
            return possible_moves[move]

        predictions = np.array(predictions)
        predictions = predictions / predictions.sum()
        move = np.random.choice(np.arange(0, len(predictions)), p=predictions)
        return possible_moves[move]

    def add_to_replay_buffer(self, state, probabilities):
        self.replay_buffer.append((state, probabilities))

    def train(self):
        minibatch = random.sample(self.replay_buffer, min(self.minibatch_size, len(self.replay_buffer)))

        self.network.train(minibatch=minibatch)
