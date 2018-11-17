
from multiprocessing import Pool

from mcts.usa_metrics import uct
from mcts.default_policies import select_random

game = None
processes = 4
pool = None


def do_rollout(args):
    global game
    mcts, state, game = args
    root = TreeNode(state)

    for i in range(mcts.simulations // processes):
        node, traversed_edges = mcts.traverse(root)
        node = mcts.expand(node)
        score = mcts.rollout(node)
        mcts.backprop(traversed_edges, score)

    return root


class MCTS:

    def __init__(self, game_, usa_metric=uct, simulations=200, default_policy=select_random, state=None):
        global game, pool
        pool = Pool(processes)
        game = game_
        state = state if state else game.get_initial_state()
        self.state = state
        self.usa_metric = usa_metric
        self.simulations = simulations
        self.default_policy = default_policy

    def set_state(self, state):
        self.state = state

    def update_state(self, move):
        self.state = game.get_outcome_state(self.state, move)

    def select_move(self, return_probabilities=False):
        """
        Selects a move by carrying out MCTS.
        :return: The selected move.
        """
        probabilities = None

        if return_probabilities:
            edge, probabilities = self.select_edge(True)
        else:
            edge = self.select_edge()
        self.update_state(edge.move)

        if return_probabilities:
            return edge.move, probabilities
        return edge.move

    def select_edge(self, return_probabilities=False):
        roots = list(pool.map(do_rollout, [(self, self.state, game)] * processes))

        moves = {}

        for root in roots:
            for edge in root.children:
                if edge.move not in moves:
                    moves[edge.move] = edge.traversals
                else:
                    moves[edge.move] += edge.traversals

        move = max(moves, key=moves.get)

        if return_probabilities:
            probs = moves.items()
            return move, probs
        return move

    def traverse(self, root):
        """
        Traverses the tree by following the tree policy.
        :param root: The root of the traversal.
        :return: A tuple with the resulting node, and a list of all traversed edges.
        """
        node = root
        traversed_edges = []

        while len(node.children):
            if node.state[1] == 0:
                traversed_edge = max(node.children.keys(), key=lambda edge: edge.q() + self.usa_metric(node, edge))
            else:
                traversed_edge = min(node.children.keys(), key=lambda edge: edge.q() - self.usa_metric(node, edge))

            traversed_edge.traversals += 1
            traversed_edges.append(traversed_edge)
            node = node.children[traversed_edge]
            node.visits += 1

        return node, traversed_edges

    def expand(self, node):
        """
        Expands the given node by generating all children.
        :param node: The node that should be expanded.
        :return: A selected child of the expanded node.
        """
        node.generate_children()
        if len(node.children):
            return next(iter(node.children.values()))
        return node

    def rollout(self, node):
        """
        Does a rollout simulation from the given node.
        :param node: The node from which a rollout simulation is to be carried out.
        :return: The resulting score from the rollout.
        """
        state = node.state
        moves = game.get_moves(state)
        while len(moves):
            selected_move = self.default_policy(state, moves)
            state = selected_move[1]
            moves = game.get_moves(state)

        return game.evaluate_state(state)

    def backprop(self, traversed_edges, score):
        """
        Carries out back propagation, updating the score of all traversed edges.
        :param traversed_edges: A list of the edges traversed during the simulation.
        :param score: The score resulting from the rollout.
        """
        for edge in traversed_edges:
            edge.score += score

class TreeEdge:
    """
    Represents a single edge in the tree.
    """
    def __init__(self, move):
        self.move = move
        self.traversals = 1
        self.score = 0

    def q(self):
        """
        Calculates the q value of the edge.
        :return: The q value.
        """
        return self.score / self.traversals

    def __hash__(self):
        return hash(self.move)


class TreeNode:
    """
    Represents a single node in the tree.
    """

    def __init__(self, state):
        self.state = state
        self.children = {}
        self.visits = 1

    def generate_children(self, n_children=-1):
        """
        Generates children of the node.
        :param n_children: Number of children to be generated. If -1, all children will be generated.
        """
        possible_moves = game.get_moves(self.state)
        n_children = len(possible_moves) if n_children == -1 else min(len(possible_moves), n_children)

        for i in range(len(self.children), n_children):
            self.children[TreeEdge(possible_moves[i][0])] = TreeNode(possible_moves[i][1])
