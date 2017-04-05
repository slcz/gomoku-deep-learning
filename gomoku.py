#!/usr/bin/env python3
import numpy as np
from enum import Enum
import random
import tensorflow as tf
from agent import (Agent, HumanAgent, RandomAgent)
from dqn import (DqntrainAgent, DqntestAgent, CloneNetworks)
import sys

class Result(Enum):
    WIN      = 0
    TIE      = 1
    CONTINUE = 2

class Gomoku:
    def __init__(self, size, connection_to_win, players):
        self.size = size
        self.win = connection_to_win
        self.clear(players)

    def clear(self, players):
        self.fst_board = np.zeros((self.size, self.size), dtype=np.bool_)
        self.snd_board = np.zeros((self.size, self.size), dtype=np.bool_)
        self.fst_str, self.snd_str = players
        self.moves = 0

    def __bounds_check(self, position):
        x, y = position
        return x < 0 or y < 0 or x >= self.size or y >= self.size

    def __check(self, board, x, y, dx, dy):
        connected = 0
        while True:
            if self.__bounds_check((x, y)):
                break
            if not board[x, y]:
                break
            connected += 1
            x, y = x - dx, y - dy
        return connected

    def occupied(self, position):
        x, y = position
        return self.fst_board[x, y] or self.snd_board[x, y]

    def check(self, position, direction):
        x, y = position
        dx, dy = direction
        if self.fst_board[x, y]:
            board = self.fst_board
        elif self.snd_board[x, y]:
            board =  self.snd_board
        connected =  self.__check(board, x, y, dx, dy)
        connected += self.__check(board, x, y, -dx, -dy)
        return connected - 1

    def move(self, position):
        x, y = position
        assert not self.fst_board[x, y] and not self.snd_board[x, y]
        self.moves += 1
        self.fst_board[x, y] = True
        for direction in ((1,0), (0,1), (1,1), (1, -1)):
            if self.check(position, direction) >= self.win:
                return Result.WIN
        if self.moves == self.size ** 2:
            return Result.TIE
        self.fst_board, self.snd_board = self.snd_board, self.fst_board
        self.fst_str, self.snd_str = self.snd_str, self.fst_str

        return Result.CONTINUE

    def print(self):
        print("  ", end="")
        for x in range(1, self.size + 1):
            print("{:2}".format(x), end="")
        print()
        for x in range(1, self.size + 1):
            print("{:2} ".format(chr(ord('a') + x - 1)), end="")
            for y in range(1, self.size + 1):
                if self.fst_board[x - 1, y - 1]:
                    c = self.fst_str
                elif self.snd_board[x - 1, y - 1]:
                    c = self.snd_str
                else:
                    c = ' '
                print("{:2}".format(c), end='')
            print()
        print()

def play_game(game, players, display):
    for player in players:
        player.agent.clear()
    game.clear((players[0].str, players[1].str))
    if display:
        game.print()
    result = Result.CONTINUE
    moves = 0
    a, b = players
    while result == Result.CONTINUE:
        move = a.agent.self_move()
        b.agent.opponent_move(move)
        a, b = b, a
        result = game.move(move)
        if display:
            game.print()
        moves += 1

    if result == Result.WIN:
        b.agent.finish(1.0)
        a.agent.finish(-1.0)
    elif result == Result.TIE:
        b.agent.finish(0.0)
        a.agent.finish(0.0)

    return moves, result

class Player:
    def __init__(self):
        self.agent = None
        self.str   = None
        self.score = 0.0
        self.score_delta = 0.0

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('agent1', 'random', "agent 1")
tf.app.flags.DEFINE_string('agent2', 'random', "agent 2")
tf.app.flags.DEFINE_boolean('board', False, "display board")
tf.app.flags.DEFINE_integer('boardsize', 9, "board size")
tf.app.flags.DEFINE_integer('connections', 5, "connected stones to win")
tf.app.flags.DEFINE_boolean('clone', False, "clone networks")

def main(argv=None):
    tf.logging.set_verbosity(tf.logging.ERROR)
    sess = tf.Session()
    if FLAGS.clone:
        clone = CloneNetworks(FLAGS.boardsize, sess)
        sess.run(tf.global_variables_initializer())
        clone.copy()
        print("Cloning done!")
        sys.exit(0)
    p1, p2 = players = (Player(), Player())
    i = 0
    for p_ in FLAGS.agent1, FLAGS.agent2:
        p = p_.capitalize() + 'Agent({}, {})'.format(FLAGS.boardsize, "sess")
        try:
            players[i].agent = eval(p)
        except (NameError, ValueError, SyntaxError):
            print("Agent {} or {} does not exist".format(FLAGS.agent1, FLAGS.agent2))
            sys.exit(1)
        i += 1
    sess.run(tf.global_variables_initializer())

    for player in players:
        player.str = player.agent.name()
    if players[0].str == players[1].str:
        players[0].str = 'X'
        players[1].str = 'O'
    game = Gomoku(FLAGS.boardsize, FLAGS.connections, (players[0].str, players[1].str))
    if FLAGS.board:
        summary_interval = 1
    else:
        summary_interval = 100

    games = 0
    games_delta = 0
    total_moves = 0
    total_moves_delta = 0
    while True:
        moves, result = play_game(game, players, FLAGS.board)
        total_moves += moves
        if FLAGS.board:
            if result == result.WIN:
                print("{} WIN".format(players[(moves - 1) % 2].str))
            else:
                print("TIE")
        if result == result.WIN:
            players[(moves - 1) % 2].score += 1.0
        else:
            players[0].score += 0.5
            players[1].score += 0.5
        a, b = players
        players = b, a
        if games > 0 and games % summary_interval == 0:
            print("* {:7}: ".format(games), end="")
            for i in players:
                score = i.score - i.score_delta
                print("{} {:8}, ".format(i.str, score), end="")
                i.score_delta = i.score
                i.agent.info()
            print("{:8}".format((total_moves - total_moves_delta) // (games - games_delta)), end="")
            print("")
            sys.stdout.flush()
            games_delta = games
            total_moves_delta = total_moves
        games += 1

if __name__ == "__main__":
    tf.app.run()