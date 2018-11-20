from games.hex import Hex
from games.random_player import RandomPlayer
from mcts.mcts import MCTS

game = Hex()
p1 = MCTS(game, simulations=50)
p2 = RandomPlayer(game)
players = (p1, p2)
games = 20

p2_starting = False
wins = 0
for i in range(games):
    state = game.get_initial_state()
    turn = p2_starting

    while not game.is_finished(state):
        for p in players:
            p.set_state(state)
        move = players[int(turn)].select_move()
        state = game.get_outcome_state(state, move)
        turn = not turn

    result = game.evaluate_state(state)
    if p2_starting and result == -1 or not p2_starting and result == 1:
        wins += 1
        print(f'Won game {i+1}')
    else:
        print(f'Lost game {i+1}')

    p2_starting = not p2_starting

print(f'Won {wins}/{games} games.')
