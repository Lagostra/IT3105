from drl.train_actor import ActorTrainer
from games.hex import Hex

class TOPP:

    def __init__(self, checkpoints, checkpoint_directory=None):
        self.game = Hex()
        self.actors = []
        for c in checkpoints:
            if type(c) == int:
                trainer = ActorTrainer(self.game, checkpoint_directory, start_game=c)
                actor = trainer.actor
            elif type(c) == tuple:
                trainer = ActorTrainer(self.game, c[0], start_game=c[1])
                actor = trainer.actor

            self.actors.append(actor)

    def play_single_game(self, p1, p2, verbose=False):
        players = (self.actors[p1], self.actors[p2])

        if verbose:
            print(f'Playing game between player {p1} and player {p2}.')

        state = self.game.get_initial_state()

        current_player = 0
        while not self.game.is_finished(state):
            move = players[current_player].select_move(state)
            state = move[1]
            current_player = (current_player + 1) % 2

        result = self.game.evaluate_state(state)

        if verbose:
            if result == 1:
                print(f'Player {p1} won')
            elif result == -1:
                print(f'Player {p2} won')
            print()

        return result

    def run_tournament(self, num_games=50, verbose=False):
        num_players = len(self.actors)
        wins = [[0] * num_players for i in range(num_players)]
        for p1 in range(num_players):
            for p2 in range(num_players):
                if p1 == p2:
                    continue

                for g in range(num_games // 2):
                    result = self.play_single_game(p1, p2, verbose)

                    if result == 1:
                        wins[p1][p2] += 1
                    elif result == -1:
                        wins[p2][p1] += 1

        return wins


if __name__ == '__main__':
    num_games = 30
    topp = TOPP([('model/test', 0)], '')
    result = topp.run_tournament(verbose=False, num_games=num_games)
    for row in result:
        for col in row:
            print(str(col).ljust(len(str(num_games)) + 2), end='')
        print(f'\t Total: {sum(row)}')
    print()
