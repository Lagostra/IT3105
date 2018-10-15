from mcts.mcts import MCTS
from games.nim import Nim
from games.random_player import RandomPlayer

game = Nim(20, 3)
player1 = MCTS(game)
player2 = RandomPlayer(game)
players = [player1, player2]


def run_single_game(starting_player=0):
    state = game.get_initial_state(starting_player)
    current_player = starting_player
    for p in players:
        p.set_state(state)

    while not game.is_finished(state):
        move = players[current_player].select_move()
        print(game.get_move_string(state, move))
        state = game.get_outcome_state(state, move)
        current_player = (current_player + 1) % 2
        if not players[0] == players[1]:
            players[current_player].update_state(move)
        score = game.evaluate_state(state)
        if score == 1:
            print('Player 1 wins!')
        elif score == -1:
            print('Player 2 wins!')


def main():
    run_single_game()


if __name__ == '__main__':
    main()
