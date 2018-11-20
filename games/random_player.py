import random


class RandomPlayer:
    """
    A player that will always pick a random move.
    """

    def __init__(self, game, state=None):
        self.state = state if state else game.get_initial_state()
        self.game = game

    def select_move(self):
        """
        Select a random move from the allowed moves.
        :return: A move, as represented by the game class.
        """
        move = self.select_move_from_state(self.state)
        self.state = self.game.get_outcome_state(self.state, move[0])
        return move[0]

    def select_move_from_state(self, state):
        move = random.choice(self.game.get_moves(state))
        return move

    def set_state(self, state):
        """
        Sets the state of the game; for example upon starting a new game.
        :param state: The new state of the game.
        """
        self.state = state

    def update_state(self, move):
        """
        Updates the state of the game by applying the given move; typically a move chosen by another player.
        :param move: The move that will be applied, as represented by the game class.
        """
        self.state = self.game.get_outcome_state(self.state, move)
