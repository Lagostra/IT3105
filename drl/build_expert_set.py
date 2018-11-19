from multiprocessing import Pool

import numpy as np

from drl.actor import Actor
from mcts.mcts import MCTS
from games.hex import Hex

games_per_series = 10
series = 10
rollouts = 5000
processes = 8
replay_save_interval = 10


def play_series(x):
    game = Hex()
    actor = Actor(game, [], replay_file='model/replays_expert.txt', rp_save_interval=replay_save_interval)
    mcts = MCTS(game, simulations=rollouts)

    for i in range(games_per_series):
        print('Starting game 1')
        state = game.get_initial_state()
        mcts.set_state(state)
        while not game.is_finished(state):
            move, probabilities = mcts.select_move(True)
            padded_probs = np.pad(probabilities, (0, game.num_possible_moves() - len(probabilities)), 'constant')
            actor.add_to_replay_buffer(state, padded_probs)
            state = game.get_outcome_state(state, move)
            mcts.set_state(state)


if __name__ == '__main__':
    pool = Pool(processes)
    for i in range(series):
        print(f'Starting series {i}')
        pool.map(play_series, [0]*8)


