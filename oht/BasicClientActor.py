from oht.BasicClientActorAbs import BasicClientActorAbs
import math
import time

from drl.actor import Actor
from games.hex import Hex
from drl.train_actor import ActorTrainer
from mcts.mcts import MCTS


class BasicClientActor(BasicClientActorAbs):
    def __init__(self, ip_address=None, verbose=True, auto_test=False):
        self.series_id = -1
        self.starting_player = -1
        self.game_count = 0
        self.series_count = 0
        self.series_game_count = 0
        BasicClientActorAbs.__init__(self, ip_address, verbose=verbose, auto_test=auto_test)

        trainer = ActorTrainer(self.hex, 'model/1000x500x100-200', start_game=250)
        #self.actor = trainer.actor
        self.actor = MCTS(self.hex, simulations=100)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current
        board. Remember to use the correct player_number for YOUR actor! The default action is to select a random empty
        cell on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        current_player = state[0] - 1
        board = list(state[1:])
        state = (board, current_player)
        #next_move = self.actor.select_move(state)[0][0]
        self.actor.set_state(state)
        next_move = self.actor.select_move()[0]
        return next_move

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id: integer identifier for the player within the whole tournament database
        :param series_id: (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map: a list of tuples: (unique-id series-id) for all players in a series
        :param num_games: number of games to be played in the series
        :param game_params: important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        self.series_count += 1
        print(f'Series {self.series_count} starting')
        print(f'Series ID: {series_id}')
        self.series_game_count = 0
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        self.game_count += 1
        print(f'Game {self.game_count} starting. (Game {self.series_game_count} in series.)')
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner
        and the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print()
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state:')
        self.print_state(end_state)

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(f'Series ID: {self.series_id}')
        for stat in stats:
            if stat[1] == self.series_id:
                # Found my stats
                print(f'Won {stat[2]}/{stat[2] + stat[3]} ({stat[2]/(stat[2]+stat[3]):.0%})')
        print()
        # print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param illegal_action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    start_time = time.time()
    bsa = BasicClientActor(verbose=False, auto_test=False)
    # bsa.handle_get_action((1,
    #                        1, 1, 1, 1, 0,
    #                        0, 0, 0, 0, 2,
    #                        0, 0, 0, 0, 2,
    #                        0, 0, 0, 0, 2,
    #                        0, 0, 0, 0, 2))
    bsa.connect_to_server()
    print(time.time() - start_time)
