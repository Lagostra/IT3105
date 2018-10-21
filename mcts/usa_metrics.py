"""
A collection of u(s,a) metrics to be used with Monte-Carlo Tree Search
"""

import math


def uct(current_node, edge, c=1):
    """
    Calculates the UCT (upper confidence bound) value of the edge from the given node.
    :param current_node: The node that we are moving from.
    :param edge: The edge that is evaluated.
    :param c: The c weighting parameter of the formula. (Optional, defaults to 1)
    :return:
    """
    return c * math.sqrt((math.log(current_node.visits))/(1 + edge.traversals))
