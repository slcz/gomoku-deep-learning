import random
import numpy as np

class Agent:
    def __init__(self, size, session):
        self.size = size
        self.session = session
    def opponent_move(self, position):
        pass
    def self_move(self):
        pass
    def finish(self, result):
        pass
    def clear(self):
        pass
    def name(self):
        return "D"
    def random_policy(self, board):
        s = self.size - 1
        move = random.randint(0, s), random.randint(0, s)
        while board[move]:
            move = random.randint(0, s), random.randint(0, s)
        return move
    def info(self):
        pass

class RandomAgent(Agent):
    def __init__(self, size, session):
        super().__init__(size, session)
        self.clear()
    def clear(self):
        self.board = np.zeros((self.size, self.size), dtype=np.bool_)
    def name(self):
        return "X"
    def opponent_move(self, position):
        x, y = position
        assert not self.board[x, y]
        self.board[x, y] = True
    def self_move(self):
        move = super().random_policy(self.board)
        self.board[move] = True
        return move

class HumanAgent(Agent):
    def __init__(self, size, session):
        super().__init__(size, session)
        self.clear()
    def clear(self):
        self.board = np.zeros((self.size, self.size), dtype=np.bool_)
    def name(self):
        return "H"
    def opponent_move(self, position):
        x, y = position
        assert not self.board[x, y]
        self.board[x, y] = True
    def self_move(self):
        x, y = None, None
        while True:
            move_str = input("enter your move e.g. a1:")
            if len(move_str) != 2:
                continue
            x = ord(move_str[0]) - ord('a')
            y = ord(move_str[1]) - ord('1')
            if x >= 0 and x < self.size and y >= 0 and y < self.size \
                    and not self.board[x, y]:
                break
        self.board[x, y] = True
        return x, y

