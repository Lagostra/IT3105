

class Nim:
    """
    Represents a game of Nim, providing methods to generate and evaluate states and moves.
    The class itself is stateless (that is, all variables simply define the variation of Nim).
    An actual game is represented by state variables passed between the methods.
    """

    def __init__(self, num_pieces, max_pieces, min_pieces=1, last_piece_wins=True):
        """
        Creates a new simple Nim game.
        :param num_pieces: Number of total pieces at the start of the game.
        :param max_pieces: The maximum number of pieces that can be removed.
        :param min_pieces: The minimum number of pieces that can be removed. Default: 1
        :param last_piece_wins: If True, the player picking up the last piece wins; if False, this player loses.
                                Default: True
        """
        self.num_pieces = num_pieces
        self.max_pieces = max_pieces
        self.min_pieces = min_pieces
        self.last_piece_wins = last_piece_wins

    def get_initial_state(self, player=0):
        """
        Returns the initial state of the game.
        :param player: The player starting the game. Default: 0
        :return: The generated game state.
        """
        return self.num_pieces, player

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
        """
        Generates the move resulting from an initial state and an applied move.
        :param initial_state: The state before the move is carried out.
        :param move: The move that should be applied.
        :return: The new state of the game.
        """
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
        """
        Generates a human-readable string representing a move.
        :param initial_state: The state of the game before the move.
        :param move: The selected move.
        :return: A string representing the move in a human-readable format.
        """
        return "Player " + str(move[1] + 1) + " selects " + str(move[0]) + " stones. Remaining stones: " + \
               str(initial_state[0] - move[0])

    def evaluate_state(self, state):
        """
        Evaluates a state. Should only be called with final states.
        :param state: The game state that should be evaluated.
        :return: The score of the game state. 1 means player 1 wins, -1 means player 2 wins.
        """
        n_pieces = state[0]
        player = state[1]

        if n_pieces == 0:
            if self.last_piece_wins and player == 1 or not self.last_piece_wins and player == 0:
                return 1
            else:
                return -1
        return 0

    def is_finished(self, state):
        """
        Returns True if the game represented by state is finished.
        :param state: The game state.
        :return: True if the game if finished, False if not.
        """
        return state[0] == 0
