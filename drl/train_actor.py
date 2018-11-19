import numpy as np
import time

from drl.actor import Actor
from mcts.mcts import MCTS
from games.hex import Hex
from games.random_player import RandomPlayer

import tensorflow as tf


def model_function(x):
    board = x[:, :-2]
    board = tf.reshape(board, (-1, 5, 5, 1))
    player = x[:, -2:]

    conv1 = tf.layers.conv2d(board, 1, 2)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    pool1_flat = tf.layers.flatten(pool1)

    x = tf.concat([pool1_flat, player], 1)
    x = tf.layers.dense(x, 50, activation=tf.nn.relu)
    x = tf.layers.dense(x, 25, activation=tf.nn.relu)
    return x


game = Hex()
layers = model_function #[50, 25]
save_interval = 25
start_game = 0
num_games = 100
rollouts = 200
checkpoint_base = 'model/conv_'
replay_file = 'model/replays_200_rollouts.txt'
replay_save_interval = 1000
one_hot_encode_state = False


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


def train():
    mcts = MCTS(game, default_policy=actor_default_policy, simulations=rollouts)
    #mcts = MCTS(game, simulations=rollouts)
    if start_game == 0:
        actor.network.save(checkpoint_base + '0.ckpt')

    for i in range(start_game, start_game + num_games):
        game_start_time = time.time()
        print("[GAME {}] Initializing state".format(i + 1))
        state = game.get_initial_state()
        mcts.set_state(state)

        print("[GAME {}] Simulating game".format(i + 1))
        # num_moves = 0
        # total_time = 0
        while not game.is_finished(state):
            # num_moves += 1
            # start_time = time.time()
            move, probabilities = mcts.select_move(return_probabilities=True)
            # total_time += time.time() - start_time
            padded_probs = np.pad(probabilities, (0, game.num_possible_moves() - len(probabilities)), 'constant')
            actor.add_to_replay_buffer(state, padded_probs)
            state = game.get_outcome_state(state, move)

        # print(f"[GAME {i+1}] Average time per move: {total_time / num_moves}")
        print("[GAME {}] Training neural network".format(i + 1))
        actor.train()

        if (i + 1) % save_interval == 0:
            print("[GAME {}] Saving neural network checkpoint".format(i + 1))
            actor.network.save(checkpoint_base + str(i + 1) + '.ckpt')
        print(f'[GAME {i+1}] Time elapsed: {time.time() - game_start_time:.2f}')
        print()


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
