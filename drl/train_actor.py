import numpy as np
import time

from drl.actor import Actor
from mcts.mcts_parallell import MCTS
from games.hex import Hex
from games.random_player import RandomPlayer


game = Hex()
layers = [1000, 100]
save_interval = 25
num_games = 1000
rollouts = 500


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
    actor = Actor(game, layers)
    #mcts = MCTS(game, default_policy=actor_default_policy, simulations=rollouts)
    mcts = MCTS(game, simulations=rollouts)
    for i in range(num_games):
        print("[GAME {}] Initializing state".format(i + 1))
        state = game.get_initial_state()
        mcts.set_state(state)

        print("[GAME {}] Simulating game".format(i + 1))
        #num_moves = 0
        #total_time = 0
        while not game.is_finished(state):
            #num_moves += 1
            #start_time = time.time()
            move, probabilities = mcts.select_move(return_probabilities=True)
            #total_time += time.time() - start_time
            padded_probs = np.pad(probabilities, (0, game.num_possible_moves() - len(probabilities)), 'constant')
            actor.add_to_replay_buffer(game.format_for_nn(state), padded_probs)
            state = game.get_outcome_state(state, move)

        #print(f"[GAME {i+1}] Average time per move: {total_time / num_moves}")
        print("[GAME {}] Training neural network".format(i + 1))
        actor.train()

        if (i + 1) % save_interval == 0:
            print("[GAME {}] Saving neural network checkpoint".format(i + 1))
            actor.network.save(f'model/game_{i+1}.ckpt')

        print()

    sim_games = 100
    starting = True
    won = sum(int(simulate_game_against_random(not starting)) for i in range(sim_games))
    print("Won {}/{} ({:.1%}) games against random player".format(won, sim_games, won/sim_games))
