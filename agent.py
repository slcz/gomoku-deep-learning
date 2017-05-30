import random
import numpy as np

class Agent:
    def __init__(self, size, session, scope, threads):
        self.size = size
        self.session = session
        self.threads = threads
        self.scope = scope
    def opponent_move(self, position, thread):
        pass
    def user_input(self, move):
        pass
    def self_move(self, thread):
        pass
    def finish(self, result, thread):
        pass
    def gameend(self):
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
    def __init__(self, size, session, scope, threads):
        super().__init__(size, session, scope, threads)
        self.clear()
    def clear(self):
        self.boards = []
        for i in range(self.threads):
            self.boards.append(np.zeros((self.size, self.size), dtype=np.bool_))
    def name(self):
        return "X"
    def opponent_move(self, position, thread):
        x, y = position
        board = self.boards[thread]
        assert not board[x, y]
        board[x, y] = True
    def self_move(self, thread):
        board = self.boards[thread]
        move = super().random_policy(board)
        board[move] = True
        return move

class HumanAgent(Agent):
    def __init__(self, size, session, scope, threads):
        assert(threads == 1)
        super().__init__(size, session, scope, threads)
        self.clear()
    def clear(self):
        self.board = np.zeros((self.size, self.size), dtype=np.bool_)
    def name(self):
        return "H"
    def opponent_move(self, position, thread):
        assert(thread == 0)
        x, y = position
        assert not self.board[x, y]
        self.board[x, y] = True
    def self_move(self, thread):
        assert(thread == 0)
        x, y = None, None
        while True:
            move_str = input("enter your move e.g. aA:")
            if len(move_str) != 2:
                continue
            x = ord(move_str[0]) - ord('a')
            y = ord(move_str[1]) - ord('A')
            if x >= 0 and x < self.size and y >= 0 and y < self.size \
                    and not self.board[x, y]:
                break
        self.board[x, y] = True
        return x, y

class WebAgent(Agent):
    def __init__(self, size, session, scope, threads):
        assert(threads == 1)
        super().__init__(size, session, scope, threads)
        self.clear()
    def clear(self):
        self.board = np.zeros((self.size, self.size), dtype=np.bool_)
        self.waiting = False
        self.nextmove = None
    def name(self):
        return "w"
    def opponent_move(self, position, thread):
        assert(thread == 0)
        x, y = position
        assert not self.board[x, y]
        self.board[x, y] = True
    def user_input(self, move):
        print(move)
        if self.waiting:
            x, y = move // self.size, move % self.size
            if x >= 0 and x < self.size and y >= 0 and y < self.size \
                    and not self.board[x, y]:
                self.nextmove = x, y
    def self_move(self, thread):
        assert(thread == 0)
        if self.nextmove != None:
            x, y = self.nextmove
            self.nextmove = None
            self.board[x, y] = True
            self.waiting = False
            return x, y
        else:
            self.waiting = True
