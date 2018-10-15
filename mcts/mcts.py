from mcts.usa_metrics import uct
from mcts.default_policies import select_random


game = None


class MCTS:

    def __init__(self, game_, usa_metric=uct, simulations=200, default_policy=select_random, state=None):
        global game
        game = game_
        state = state if state else game.get_initial_state()
        self.root = TreeNode(state)
        self.usa_metric = usa_metric
        self.simulations = simulations
        self.default_policy = default_policy

    def set_state(self, state):
        self.root = TreeNode(state)

    def update_state(self, move):
        """
        Updates state based on move selected by opponent.
        """
        self.root = TreeNode(game.get_outcome_state(self.root.state, move))

    def select_move(self):
        edge = self.select_edge()
        self.root = self.root.children[edge]
        self.root.children = {}
        self.root.visits = 1

        return edge.move

    def select_edge(self):
        for i in range(self.simulations):
            node, traversed_edges = self.traverse(self.root)
            node = self.expand(node)
            score = self.rollout(node)

            for edge in traversed_edges:
                edge.score += score

        return max(self.root.children.keys(), key=lambda e: e.traversals)

    def traverse(self, root):
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
        node.generate_children()
        if len(node.children):
            return next(iter(node.children.values()))
        return node

    def rollout(self, node):
        state = node.state
        moves = game.get_moves(state)
        while len(moves):
            selected_move = self.default_policy(state, moves)
            state = selected_move[1]
            moves = game.get_moves(state)

        return game.evaluate_state(state)


class TreeEdge:
    def __init__(self, move):
        self.move = move
        self.traversals = 1
        self.score = 0

    def q(self):
        return self.score / self.traversals

    def __hash__(self):
        return hash(self.move)


class TreeNode:

    def __init__(self, state):
        self.state = state
        self.children = {}
        self.visits = 1

    def generate_children(self, n_children=-1):
        possible_moves = game.get_moves(self.state)
        n_children = len(possible_moves) if n_children == -1 else min(len(possible_moves), n_children)

        for i in range(len(self.children), n_children):
            self.children[TreeEdge(possible_moves[i][0])] = TreeNode(possible_moves[i][1])
