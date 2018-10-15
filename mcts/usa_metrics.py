import math


def uct(current_node, edge, c=1):
    return c * math.sqrt((math.log(current_node.visits))/(1 + edge.traversals))
