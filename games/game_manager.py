from mcts.mcts import MCTS
from games.nim import Nim
from games.random_player import RandomPlayer

game = Nim(20, 3)
player1 = MCTS(game)
player2 = RandomPlayer(game)
players = [player1, player2]


def run_single_game(starting_player=0, verbose=False):
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


def main(verbose=False, play_mode='mix', num_games=50):
    wins = 0
    starting_player = play_mode if type(play_mode) == int else 0
    for i in range(num_games):
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

        if play_mode == 'mix':
            starting_player = (starting_player + 1) % 2
    print('Player 1 won {} out of {} games; i.e. {:.1f}% of the played games.'.format(wins,
                                                                                      num_games, (wins/num_games)*100))


if __name__ == '__main__':
    main(num_games=50, verbose=True)
