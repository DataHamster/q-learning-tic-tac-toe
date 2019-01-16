import numpy as np
import collections
import time

Gamma = 0.9
Alpha = 0.2


class Environment:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.x = -1  # player with an x
        self.o = 1  # player with an o
        self.winner = None
        self.ended = False
        self.actions = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1),
                        5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}

    def reset_env(self):
        self.board = np.zeros((3, 3))
        self.winner = None
        self.ended = False

    def reward(self, sym):
        if not self.game_over():
            return 0
        if self.winner == sym:
            return 10
        else:
            return 0

    def get_state(self,):
        k = 0
        h = 0
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3**k) * v
                k += 1
        return h

        def random_action(self):
            return np.random.choice(self.actions.keys())

    def make_move(self, player, action):
        i, j = self.actions[action]
        if self.board[i, j] == 0:
            self.board[i, j] = player

    def game_over(self, force_recalculate=False):
        # returns true if game over (a player has won or it's a draw)
        # otherwise returns false
        # also sets 'winner' instance variable and 'ended' instance variable
        if not force_recalculate and self.ended:
            return self.ended

        # check rows
        for i in range(3):
            for player in (self.x, self.o):
                if self.board[i].sum() == player*3:
                    self.winner = player
                    self.ended = True
                    return True

        # check columns
        for j in range(3):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player*3:
                    self.winner = player
                    self.ended = True
                    return True

        # check diagonals
        for player in (self.x, self.o):
            # top-left -> bottom-right diagonal
            if self.board.trace() == player*3:
                self.winner = player
                self.ended = True
                return True
            # top-right -> bottom-left diagonal
            if np.fliplr(self.board).trace() == player*3:
                self.winner = player
                self.ended = True
                return True

        # check if draw
        if np.all((self.board == 0) == False):
            # winner stays None
            self.winner = None
            self.ended = True
            return True

        # game is not over
        self.winner = None
        return False

    def draw_board(self):
        for i in range(3):
            print("-------------")
            for j in range(3):
                print("  ", end="")
                if self.board[i, j] == self.x:
                    print("x ", end="")
                elif self.board[i, j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")


class Agent:
    def __init__(self, Environment, sym):
        self.q_table = collections.defaultdict(float)
        self.env = Environment
        self.epsylon = 1.0
        self.sym = sym
        self.ai = True

    def best_value_and_action(self, state):
        best_val, best_act = None, None
        for action in self.env.actions.keys():
            action_value = self.q_table[(state, action)]
            if best_val is None or best_val < action_value:
                best_val = action_value
                best_act = action
        return best_val, best_act

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + Gamma * best_v
        old_val = self.q_table[(s, a)]
        self.q_table[(s, a)] = old_val * (1-Alpha) + new_val * Alpha

    def play_step(self, state, random=True):
        if random == False:
            epsylon = 0
        cap = np.random.rand()
        if cap > self.epsylon:
            _, action = self.best_value_and_action(state)
        else:
            action = np.random.choice(list(self.env.actions.keys()))
            self.epsylon *= 0.99998
        self.env.make_move(self.sym, action)
        new_state = self.env.get_state()
        if new_state == state and not self.env.ended:
            reward = -5
        else:
            reward = self.env.reward(self.sym)
        self.value_update(state, action, reward, new_state)


class Human:
    def __init__(self, env, sym):
        self.sym = sym
        self.env = env
        self.ai = False

    def play_step(self):
        while True:
            move = int(input('enter position like: \n0|1|2\n------\n3|4|5\n------\n6|7|8'))
            if move in list(self.env.actions.keys()):
                break
        self.env.make_move(self.sym, move)


def main():
    env = Environment()
    p1 = Agent(env, env.x)
    p2 = Agent(env, env.o)
    draw = 1
    for t in range(1000005):

        current_player = None
        episode_length = 0
        while not env.game_over():
            # alternate between players
            # p1 always starts first
            if current_player == p1:
                current_player = p2
            else:
                current_player = p1

            # current player makes a move
            current_player.play_step(env.get_state())

        env.reset_env()

        if t % 1000 == 0:
            print(t)
            print(p1.q_table[(0, 0)])
            print(p1.q_table[(0, 1)])
            print(p1.q_table[(0, 2)])
            print(p1.q_table[(0, 3)])
            print(p1.q_table[(0, 4)])
            print(p1.q_table[(0, 5)])
            print(p1.q_table[(0, 6)])
            print(p1.q_table[(0, 7)])
            print(p1.q_table[(0, 8)])
            print(p1.epsylon)

    env.reset_env()
    # p1.sym = env.x

    while True:
        while True:
            first_move = input("Do you want to make the first move? y/n :")
            if first_move.lower() == 'y':
                first_player = Human(env, env.x)
                second_player = p2
                break
            else:
                first_player = p1
                second_player = Human(env, env.o)
                break
        current_player = None

        while not env.game_over():
            # alternate between players
            # p1 always starts first
            if current_player == first_player:
                current_player = second_player
            else:
                current_player = first_player
            # draw the board before the user who wants to see it makes a move

            if current_player.ai == True:
                current_player.play_step(env.get_state(), random=False)
            if current_player.ai == False:
                current_player.play_step()
            env.draw_board()
        env.draw_board()
        play_again = input('Play again? y/n: ')
        env.reset_env()
        # if play_again.lower != 'y':
        #     break


if __name__ == "__main__":
    main()
