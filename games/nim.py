

class Nim:

    def __init__(self, num_pieces, max_pieces, min_pieces=1, last_piece_wins=True):
        self.num_pieces = num_pieces
        self.max_pieces = max_pieces
        self.min_pieces = min_pieces
        self.last_piece_wins = last_piece_wins

    def get_initial_state(self):
        return [self.num_pieces, 0]

    def get_next_states(self, state):
        n_pieces = state[0]
        player = state[1]

        states = []
        for pieces_removed in range(self.min_pieces, self.max_pieces + 1):
            if n_pieces - pieces_removed < 0:
                break

            states.append([n_pieces - pieces_removed, int(not player)])

        return states

    def evaluate_state(self, state):
        n_pieces = state[0]
        player = state[1]

        if n_pieces == 0:
            if self.last_piece_wins and player == 0 or not self.last_piece_wins and player == 1:
                return 1
            else:
                return -1
        return 0
