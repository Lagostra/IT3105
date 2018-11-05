
class Hex:

    def __init__(self, size=4):
        self.size = size

    def get_initial_state(self):
        return [0] * (self.size**2)

    def get_moves(self, state):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if state[i * self.size + j] == 0:
                    # TODO Add move
                    pass
        return moves

    def get_outcome_state(self, initial_state, move):
        pass

    def get_move_string(self, inital_state, move):
        pass

    def get_state_string(self, state):
        newsize = self.size * 2 - 1
        grid = [[" " for x in range(newsize)] for y in range(newsize)]
        for i in range(self.size):
            for j in range(self.size):
                y = i + j
                x = (self.size - 1) + j - i
                grid[y][x] = state[i * self.size + j]
        return '\n'.join(["".join(str(item) for item in row) for row in grid])

    def unflatten(self, state):
        for i in range(0, len(state), self.size):
            yield state[i:i + self.size]

    def _get_neighbours(self, slot):
        y = slot[0]
        x = slot[1]
        neighbours = []
        if y > 0:
            if x > 0:
                neighbours.append((y-1, x-1))
            neighbours.append((y-1, x))
            if x + 1 < self.size:
                neighbours.append((y-1, x+1))
        if y + 1 < self.size:
            if x > 0:
                neighbours.append((y+1, x-1))
            neighbours.append((y+1, x))
            if x + 1 < self.size:
                neighbours.append((y+1, x+1))
        if x > 0:
            neighbours.append((y, x-1))
        if x + 1 < self.size:
            neighbours.append((y, x+1))
        return neighbours

    def evaluate_state(self, state):
        def is_won(pos, player, visited=[]):
            visited.append(pos)
            if player == 1 and pos[0] == self.size - 1\
                    or player == 2 and pos[1] == self.size - 1:
                return True

            for neighbour in self._get_neighbours(pos):
                if state[neighbour[0] * self.size + neighbour[1]] == player\
                        and neighbour not in visited:
                    result = is_won(neighbour, player, visited)
                    if result:
                        return True
            visited.remove(pos)
            return False

        for i in range(self.size):
            if is_won((0, i), 1):
                return 1
            if is_won((i, 0), 2):
                return 2

        return 0

    def is_finished(self, state):
        return self.evaluate_state(state) != 0

    def num_possible_moves(self):
        return self.size**2

    def state_size(self, one_hot_encoded=True):
        if one_hot_encoded:
            return self.size**2 * 2
        return self.size**2


if __name__ == '__main__':
    state = [
        1, 0, 0, 0,
        0, 0, 0, 0,
        1, 2, 2, 2,
        2, 2, 0, 0
    ]

    hex = Hex()

    print(hex.evaluate_state(state))

