import time
from collections import deque
import itertools
import random

import numpy as np

from drl.actor import Actor
from mcts.mcts import MCTS
from games.hex import Hex
from games.random_player import RandomPlayer


class ActorTrainer:

    def __init__(self, game, actor, checkpoint_directory, network_save_interval=100, rollouts=100,
                 start_game=0, replay_save_interval=250, replay_limit=20000, minibatch_size=50,
                 replay_file=None):
        self.game = game
        self.actor = actor
        self.checkpoint_directory = checkpoint_directory
        self.network_save_interval = network_save_interval
        self.mcts = MCTS(game, simulations=rollouts, default_policy=self.create_default_policy())
        self.game_count = start_game
        self.replay_save_interval = replay_save_interval
        self.replay_buffer = deque(maxlen=replay_limit)
        self.rp_count = 0
        self.minibatch_size = minibatch_size
        self.replay_file = replay_file

        if replay_save_interval > replay_limit:
            raise ValueError(f'replay_save_interval ({replay_save_interval}) must be smaller '
                             f'than replay_limit ({replay_limit})')

        if replay_file is not None:
            try:
                self.load_replays()
            except FileNotFoundError:
                pass

        if start_game == 0:
            actor.save_checkpoint(checkpoint_directory + '/game_0')

    def train(self, num_games):
        for i in range(num_games):
            self.game_count += 1
            game_start_time = time.time()
            print(f'[GAME {self.game_count} Initializing state')
            state = self.game.get_initial_state()
            self.mcts.set_state(state)

            print(f'[GAME {self.game_count} Simulating game')
            while not self.game.is_finished(state):
                move, probabilities = self.mcts.select_move(True)
                padded_probs = np.pad(probabilities, (0, self.game.num_possible_moves() - len(probabilities)),
                                      'constant')
                self.add_to_replay_buffer(state, padded_probs)
                state = game.get_outcome_state(state, move)
                self.mcts.set_state(state)

            print(f'[GAME {self.game_count}] Training neural network')
            self.train_network()

            if self.game_count % self.network_save_interval == 0:
                print(f'[GAME {self.game_count}] Saving neural network checkpoint')
                self.actor.save_checkpoint(f'{self.checkpoint_directory}/game_{self.game_count}')
            print(f'[GAME {self.game_count}] Time elapsed: {time.time() - game_start_time:.2f}')
            print()

    def train_network(self):
        minibatch = random.sample(self.replay_buffer, min(self.minibatch_size, len(self.replay_buffer)))
        for i in range(len(minibatch)):
            minibatch[i] = self.game.format_for_nn(minibatch[i][0]), minibatch[i][1]
        self.actor.network.train(minibatch=minibatch)

    def create_default_policy(self):
        def actor_default_policy(state, moves):
            move = self.actor.select_move(state, stochastic=True)
            return move

        return actor_default_policy

    def add_to_replay_buffer(self, state, probabilities):
        self.replay_buffer.append((state, probabilities))
        self.rp_count += 1

        if self.replay_save_interval != -1 and self.rp_count % self.replay_save_interval == 0 and self.rp_count != 0:
            replays = len(self.replay_buffer)
            self.save_replays(itertools.islice(self.replay_buffer, replays - self.replay_save_interval, replays))

    def save_replays(self, replays):
        if self.replay_file is None:
            return

        with open(self.replay_file, 'a') as f:
            for replay in replays:
                state_string = ','.join(map(str, replay[0][0])) + ',' + str(replay[0][1])
                probs_string = ','.join(map(str, replay[1]))
                rp_string = state_string + ';' + probs_string
                f.write(rp_string + '\n')

    def load_replays(self):
        with open(self.replay_file, 'r') as f:
            for line in f:
                state, probs = line.split(';')
                state = list(map(int, state.split(',')))
                player = state[-1]
                board = state[:-1]
                probs = list(map(float, probs.split(',')))
                self.replay_buffer.append(((board, player), probs))


game = Hex()
layers = [100, 50]
save_interval = 50
start_game = 0
num_games = 200
rollouts = 100
checkpoint_base = 'model/r100_'
replay_file = 'model/replays_100_rollouts.txt'
replay_save_interval = 250
one_hot_encode_state = True


def actor_default_policy(state, moves):
    move = actor.select_move(state, stochastic=True)
    return move


def simulate_game_against_random(starting=True):
    state = game.get_initial_state()
    random_player = RandomPlayer(game)
    while not game.is_finished(state):
        if starting and state[1] == 0 or not starting and state[1] == 1:
            move = actor.select_move(state)
        else:
            move = random_player.select_move_from_state(state)
        state = move[1]

    result = game.evaluate_state(state)
    return starting and result == 1 or not starting and result == -1

if __name__ == '__main__':


    if start_game > 0:
        actor = Actor(game, layers, checkpoint=checkpoint_base + str(start_game) + '.ckpt',
                      replay_file=replay_file, rp_save_interval=replay_save_interval,
                      one_hot_encode_state=one_hot_encode_state)
    actor = Actor(game, layers, replay_file=replay_file, rp_save_interval=replay_save_interval,
                  one_hot_encode_state=one_hot_encode_state)


    train()
    # actor.network.save('model/test_0.ckpt')
    # for i in range(1000):
    #     if i % 10 == 0:
    #         print(i)
    #     actor.train()
    # actor.network.save('model/test.ckpt')

    sim_games = 100
    starting = True
    won = sum(int(simulate_game_against_random(not starting)) for i in range(sim_games))
    print("Won {}/{} ({:.1%}) games against random player".format(won, sim_games, won/sim_games))
