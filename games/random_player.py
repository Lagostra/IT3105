import random


class RandomPlayer:

    def __init__(self, game, state=None):
        self.state = state if state else game.get_initial_state()
        self.game = game

    def select_move(self):
        move = random.choice(self.game.get_moves(self.state))[0]
        self.state = self.game.get_outcome_state(self.state, move)
        return move

    def update_state(self, move):
        self.state = self.game.get_outcome_state(self.state, move)
