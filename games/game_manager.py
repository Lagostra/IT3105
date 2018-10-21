from mcts.mcts import MCTS
from games.nim import Nim
from games.random_player import RandomPlayer
import random

# Instantiate our game with given parameters
game = Nim(9, 3)
# Create a new MCTS player for player 1
player1 = MCTS(game, simulations=1000)
# Create player 2 - either as the same player as player 1, or as a random player
player2 = player1 #RandomPlayer(game)
players = [player1, player2]


def run_single_game(starting_player=0, verbose=False):
    """
    Runs a simulation of a single game, and returns the winning player.
    :param starting_player: The player that should start the game.
    :param verbose: If True, string representations of all moves will be printed to the console.
    :return: 0 if player 1 is the winner, 1 if player 2 is the winner.
    """
    state = game.get_initial_state(starting_player)
    current_player = starting_player
    for p in players:
        p.set_state(state)

    while not game.is_finished(state):
        move = players[current_player].select_move()
        if verbose:
            print(game.get_move_string(state, move))
        state = game.get_outcome_state(state, move)
        current_player = (current_player + 1) % 2
        if not players[0] == players[1]:
            players[current_player].update_state(move)
        score = game.evaluate_state(state)
        if score == 1:
            return 0
        elif score == -1:
            return 1


def main(verbose=False, play_mode='alternate', num_games=50):
    """
    Runs a number of game simulations, and prints the result.
    :param verbose: If True, string representations of all moves and winner of each individual game will be printed.
    :param play_mode: Either an int that indicates the player that will start all games, 'mix' for picking a random
                        player to start each game, and 'alternate' to make the players start every other game.
    :param num_games: Number of games to be simulated.
    """
    wins = 0
    starting_player = play_mode if type(play_mode) == int else 0
    for i in range(num_games):
        if play_mode == 'mix':
            starting_player = random.randint(0, 1)
        result = run_single_game(starting_player=starting_player, verbose=verbose)
        if result == 0:
            if verbose:
                print('Player 1 wins!')
            wins += 1
        else:
            if verbose:
                print('Player 2 wins!')
        if verbose:
            print()

        if play_mode == 'alternate':
            starting_player = (starting_player + 1) % 2
    print('Player 1 won {} out of {} games; i.e. {:.1f}% of the played games.'.format(wins,
                                                                                      num_games, (wins/num_games)*100))


if __name__ == '__main__':
    main(num_games=50, verbose=True, play_mode=0)
