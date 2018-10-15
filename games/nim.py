

class Nim:

    def __init__(self, num_pieces, max_pieces, min_pieces=1, last_piece_wins=True):
        self.num_pieces = num_pieces
        self.max_pieces = max_pieces
        self.min_pieces = min_pieces
        self.last_piece_wins = last_piece_wins

    def get_initial_state(self):
        return self.num_pieces, 0

    def get_moves(self, state):
        """
        Returns a list of possible moves given a state. Should always be given in same order.
        :param state: The current state of the game.
        :return: A list of moves and outcome states. Each item is a tuple consisting of a tuple representing the move,
                and a tuple representing the outcome state. The move has the form
                (num_pieces_removed, player_making_move), while the state has the form
                (num_pieces_remaining, player_in_the_move).
        """
        n_pieces = state[0]
        player = state[1]

        moves = []
        for pieces_removed in range(self.min_pieces, self.max_pieces + 1):
            if n_pieces - pieces_removed < 0:
                break

            moves.append(((pieces_removed, player), (n_pieces - pieces_removed, int(not player))))

        return moves

    def get_outcome_state(self, initial_state, move):
        n_pieces = initial_state[0]
        player = initial_state[1]
        n_pieces_removed = move[0]
        player_moving = move[1]

        if player != player_moving:
            raise Exception('Player ' + move[1] + ' is not in the move in the given state!')
        if n_pieces - n_pieces_removed < 0:
            raise Exception('Removing ' + n_pieces_removed + ' from the current state (with ' + n_pieces +
                            'remaining) is illegal!')
        if n_pieces_removed < self.min_pieces or n_pieces_removed > self.max_pieces:
            raise Exception('The number of removed pieces is outside the allowed range!')
        return n_pieces - n_pieces_removed, int(not player)

    def get_move_string(self, initial_state, move):
        return "Player " + move[1] + " selects " + move[0] + " stones. Remaining stones: " + initial_state[0] - move[0]

    def evaluate_state(self, state):
        n_pieces = state[0]
        player = state[1]

        if n_pieces == 0:
            if self.last_piece_wins and player == 0 or not self.last_piece_wins and player == 1:
                return 1
            else:
                return -1
        return 0
