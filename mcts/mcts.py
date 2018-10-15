from mcts.usa_metrics import uct
from mcts.default_policies import select_random


game = None


class MCTS:

    def __init__(self, game_, usa_metric=uct, simulations=200, default_policy=select_random()):
        global game
        game = game_
        self.root = TreeNode(game.get_initial_state())
        self.usa_metric = usa_metric
        self.simulations = simulations
        self.default_policy = default_policy

    def select_move(self):
        edge = self.select_edge()
        self.root = self.root.children[edge]
        self.root.children = {}

        return edge.move

    def select_edge(self):
        for i in range(self.simulations):
            node = self.traverse(self.root)
            node, traversed_edges = self.expand(node)
            score = self.rollout(node)

            for edge in traversed_edges:
                edge.score += score

        return max(self.root.children.keys(), key=lambda e: e.traversals)

    def traverse(self, root):
        node = root
        traversed_edges = []

        while len(node.children):
            if node[1] == 0:
                traversed_edge = max(node.children.keys, lambda edge: edge.q() + self.usa_metric(node, edge))
            else:
                traversed_edge = min(node.children.keys, lambda edge: edge.q() - self.usa_metric(node, edge))

            traversed_edge.traversals += 1
            traversed_edges.append(traversed_edge)
            node = node.children[traversed_edge]
            node.visits += 1

        return node, traversed_edges

    def expand(self, node):
        node.generate_children()
        return next(iter(node.children.values()))

    def rollout(self, node):
        state = node.state
        moves = game.get_moves(state)
        while len(moves):
            selected_move = self.default_policy(state, moves)

        return game.evaluate_state(selected_move[1])


class TreeEdge:
    def __init__(self, move):
        self.move = move
        self.traversals = 0
        self.score = 0

    def q(self):
        return self.score / self.traversals

    def __hash__(self):
        return hash(self.move)


class TreeNode:

    def __init__(self, state):
        self.state = state
        self.children = {}
        self.visits = 0

    def generate_children(self, n_children=-1):
        possible_moves = game.get_moves(self.state)
        n_children = len(possible_moves) if n_children == -1 else min(len(possible_moves), n_children)

        for i in range(len(self.children), n_children):
            self.children[TreeEdge(possible_moves[0])] == TreeNode(possible_moves[1])
