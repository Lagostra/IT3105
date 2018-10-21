"""
A collection of default policies to be used with Monte-Carlo Tree Search.
"""

import random


def select_random(state, moves):
    """
    Selects a random move from the list of possible moves.
    :param state: The current state of the game (unused).
    :param moves: The list of allowed moves
    :return: A randomly chosen move.
    """
    return random.choice(moves)