#!/usr/bin/env python3
import numpy as np
from enum import Enum
import random
import tensorflow as tf
from agent import (Agent, HumanAgent, RandomAgent, WebAgent)
from dqn import (DqntrainAgent, DqntestAgent, MontecarloAgent, CloneNetworks)
import sys
from collections import deque
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from rules import Rules

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('web', False, "web interface")
tf.app.flags.DEFINE_string('agent1', 'random', "agent 1")
tf.app.flags.DEFINE_string('agent2', 'random', "agent 2")
tf.app.flags.DEFINE_boolean('board', False, "display board")
tf.app.flags.DEFINE_integer('boardsize', 9, "board size")
tf.app.flags.DEFINE_integer('connections', 5, "connected stones to win")
tf.app.flags.DEFINE_boolean('clone', False, "clone networks")
tf.app.flags.DEFINE_integer('concurrency', 1, "concurrency")
tf.app.flags.DEFINE_float('stop_rate', 0.95, """win rate of agent 1 threshold to stop training""")
tf.app.flags.DEFINE_integer('max_games', 1000000, """maximum number of games""")
tf.app.flags.DEFINE_integer('min_games',  20000, """minimum number of games""")
tf.app.flags.DEFINE_integer('check_stop', 4000, """interval to check stop condition""")

class Result(Enum):
    WIN      = 0
    TIE      = 1
    CONTINUE = 2

class WebSessionState(Enum):
    START    = 0
    INGAME   = 1
    ENDGAME  = 2

class Gomoku:
    def __init__(self, size, connection_to_win, players):
        self.size = size
        self.win = connection_to_win
        self.clear(players)
        self.rules = Rules(size, self.win)

    def clear(self, players):
        self.fst_board = np.zeros((self.size, self.size), dtype=np.bool_)
        self.snd_board = np.zeros((self.size, self.size), dtype=np.bool_)
        self.fst_str, self.snd_str = players
        self.moves = 0

    def occupied(self, position):
        x, y = position
        return self.fst_board[x, y] or self.snd_board[x, y]

    def move(self, position):
        x, y = position
        assert not self.fst_board[x, y] and not self.snd_board[x, y]
        self.moves += 1
        self.fst_board[x, y] = True
        win, connections = self.rules.check_win(position, self.fst_board)
        if win:
                return Result.WIN, connections
        if self.moves == self.size ** 2:
            return Result.TIE, []
        self.fst_board, self.snd_board = self.snd_board, self.fst_board
        self.fst_str, self.snd_str = self.snd_str, self.fst_str

        return Result.CONTINUE, None

    def print(self):
        print("  ", end="")
        for x in range(1, self.size + 1):
            print("{:2}".format(chr(ord('A') + x - 1)), end="")
        print()
        for x in range(1, self.size + 1):
            print("{:2}".format(chr(ord('a') + x - 1)), end="")
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

def play_game(games, players, display):
    for player in players:
        player.agent.clear()
    for game in games:
        game.clear((players[0].str, players[1].str))
        if display:
            game.print()
    nr = len(games)
    results = []
    moves = []
    game_players = []
    for i in range(nr):
        results.append(Result.CONTINUE)
        moves.append(0)
        game_players.append(players)
    done = 0
    while done < nr:
        for i, game in enumerate(games):
            if results[i] == Result.CONTINUE:
                a, b = game_players[i]
                move = a.agent.self_move(i)
                b.agent.opponent_move(move, i)
                game_players[i] = b, a
                a, b = game_players[i]
                results[i], _ = game.move(move)
                if display:
                    game.print()
                moves[i] += 1

                if results[i] == Result.WIN:
                    b.agent.finish(1.0, i)
                    a.agent.finish(-1.0, i)
                    done += 1
                elif results[i] == Result.TIE:
                    b.agent.finish(0.0, i)
                    a.agent.finish(0.0, i)
                    done += 1

    return moves, results

class Player:
    def __init__(self):
        self.agent = None
        self.str   = None
        self.score = 0.0
        self.score_delta = 0.0

app = Flask(__name__, static_url_path="", static_folder="static")

web_context = None

def web_context_init(games, players):
    global web_context
    assert(web_context == None)
    web_context = {}
    web_context['state' ] = WebSessionState.START
    web_context['players'] = players
    web_context['game'] = games[0]

@app.route('/info')
def info():
    global web_context
    r = web_context['game'].size
    fst = web_context['game'].fst_board.ravel().tolist()
    snd = web_context['game'].snd_board.ravel().tolist()
    moves = web_context['game'].moves
    return jsonify({"result": "accepted", "size": r, "fst": fst, "snd": snd, "moves": moves})

@app.route('/next')
def next_state():
    global web_context
    if request.method != 'GET':
        return
    if web_context['state'] == WebSessionState.START:
        players = web_context['players']
        web_context['game_players'] = players
        for player in players:
            player.agent.clear()
        game = web_context['game']
        game.clear((players[0].str, players[1].str))
        web_context['state'] = WebSessionState.INGAME
        r = {"result": "clear"}
        return jsonify(r)
    elif web_context['state'] == WebSessionState.INGAME:
        players = web_context['game_players']
        game = web_context['game']
        a, b = players
        move = a.agent.self_move(0)
        if move == None:
            r = {"result": "none"}
            return jsonify(r)
        b.agent.opponent_move(move, 0)
        web_context['game_players'] = b, a
        result, connections = game.move(move)
        if result == Result.WIN:
            web_context['state'] = WebSessionState.ENDGAME
            web_context['connections'] = connections
        elif result == Result.TIE:
            web_context['state'] = WebSessionState.ENDGAME
            web_context['connections'] = []
        x, y = move
        m = int(x), int(y)
        r = {"result": "move", "move": m}
        return jsonify(r)
    elif web_context['state'] == WebSessionState.ENDGAME:
        web_context['state'] = WebSessionState.START
        a, b = web_context['players']
        web_context['players'] = b, a
        r = {"result": "end", "highlights": web_context['connections']}
        return jsonify(r)

@app.route('/move')
def user_move():
    if request.method != 'GET':
        return
    move = request.args.get('move')
    players = web_context['game_players']
    player = players[0]
    player.agent.user_input(int(move))
    r = {"result": "accepted"}
    return jsonify(r)

@app.route('/')
def mainpage():
    return render_template("interface.html")

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
        p = p_.capitalize() + 'Agent({}, {}, {})'.format(FLAGS.boardsize,
                "sess", FLAGS.concurrency)
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
    games = []
    for i in range(FLAGS.concurrency):
        games.append(Gomoku(FLAGS.boardsize,
            FLAGS.connections, (players[0].str, players[1].str)))
    if FLAGS.web:
        web_context_init(games, players)
        app.run(host="0.0.0.0", port=5000)
        return
    if FLAGS.board:
        summary_interval = 1
    else:
        summary_interval = 1024

    latest_games = deque()
    nr_games = 0
    nr_games_delta = 0
    total_moves = 0
    total_moves_delta = 0
    nr = len(games)
    threshold = False
    while nr_games < FLAGS.min_games or \
          (nr_games < FLAGS.max_games and not threshold):
        allmoves, results = play_game(games, players, FLAGS.board)
        for moves in allmoves:
            total_moves += moves
        if FLAGS.board:
            for result, moves in zip(results, allmoves):
                if result == result.WIN:
                    print("{} WIN".format(players[(moves - 1) % 2].str))
                else:
                    print("TIE")
        for result, moves in zip(results, allmoves):
            if result == result.WIN:
                players[(moves - 1) % 2].score += 1.0
                if players[(moves - 1) % 2].agent.name() == "@":
                    latest_games.append(1.0)
                else:
                    latest_games.append(0.0)
            else:
                players[0].score += 0.5
                players[1].score += 0.5
                latest_games.append(0.5)
        if len(latest_games) > FLAGS.check_stop:
            while len(latest_games) > FLAGS.check_stop:
                latest_games.popleft()
            if sum(latest_games) / FLAGS.check_stop >= FLAGS.stop_rate:
                threshold = True
        a, b = players
        players = b, a
        if nr_games > 0 and nr_games % (summary_interval / nr * nr) == 0:
            print("* {:7}: ".format(nr_games), end="")
            for i in players:
                score = i.score - i.score_delta
                print("{} {:8}, ".format(i.str, score), end="")
                i.score_delta = i.score
                i.agent.info()
            print("{:8}".format((total_moves - total_moves_delta) // \
                    (nr_games - nr_games_delta)), end="")
            print("")
            sys.stdout.flush()
            nr_games_delta = nr_games
            total_moves_delta = total_moves
        nr_games += nr
    print("* {:7}: ".format(nr_games), end="")
    for i in players:
        score = i.score - i.score_delta
        print("{} {:8}, ".format(i.str, score), end="")
        i.score_delta = i.score
        i.agent.info()
    if nr_games_delta == 0:
        nr_games_delta = 1
    print("{:8}".format((total_moves - total_moves_delta) // \
            (nr_games - nr_games_delta)), end="")
    print("")
    sys.stdout.flush()

    for player in players:
        player.agent.gameend()

if __name__ == "__main__":
    tf.app.run()
