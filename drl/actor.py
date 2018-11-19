from collections import deque
import itertools
import numpy as np
import random

from nn.network import Network


class Actor:

    def __init__(self, game, layers=[], minibatch_size=50, checkpoint=None, replay_file=None, rp_save_interval=1000,
                 one_hot_encode_state=True):
        self.game = game
        self.replay_buffer = deque(maxlen=20000)
        self.minibatch_size = minibatch_size
        self.replay_file = replay_file
        self.rp_save_interval = min(rp_save_interval, 20000)
        self.rp_count = 0
        self.one_hot_encode_state = one_hot_encode_state

        if replay_file is not None:
            try:
                self.load_replays()
            except FileNotFoundError:
                pass

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
            self.network.load(checkpoint)

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

    def add_to_replay_buffer(self, state, probabilities):
        formatted_state = self.game.format_for_nn(state, one_hot_encoded=self.one_hot_encode_state)
        self.replay_buffer.append((state, probabilities))
        self.rp_count += 1

        if self.rp_save_interval != -1 and self.rp_count % self.rp_save_interval == 0 and self.rp_count != 0:
            replays = len(self.replay_buffer)
            self.save_replays(itertools.islice(self.replay_buffer, replays - self.rp_save_interval, replays))

    def save_replays(self, replays):
        if self.replay_file is None:
            return

        with open(self.replay_file, 'a') as f:
            for replay in replays:
                rp_string = ','.join(map(str, replay[0])) + ';' + ','.join(map(str, replay[1]))
                f.write(rp_string + '\n')

    def load_replays(self):
        with open(self.replay_file, 'r') as f:
            for line in f:
                state, probs = line.split(';')
                state = list(map(int, state.split(',')))
                probs = list(map(float, probs.split(',')))
                self.replay_buffer.append((state, probs))

    def train(self):
        minibatch = random.sample(self.replay_buffer, min(self.minibatch_size, len(self.replay_buffer)))
        self.network.train(minibatch=minibatch)
