
################################################################################
#
# Minimax Q learning
#
################################################################################

from agent import (Agent, RandomAgent)
import tensorflow as tf
import numpy as np
import os
from collections import deque
import random

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', './saved_models',
        """Directory to save the trained models""")
tf.app.flags.DEFINE_string('train_generation', None,
        """train-model generation""")
tf.app.flags.DEFINE_string('test_generation', None,
        """test-model generation""")
tf.app.flags.DEFINE_float('learn_rate', 0.0005, """learning rate""")
tf.app.flags.DEFINE_string('copy_from', None, """copy from model""")
tf.app.flags.DEFINE_string('copy_to', None, """copy to model""")
tf.app.flags.DEFINE_float('train_epsilon', 0.01, """epsilon greedy""")
tf.app.flags.DEFINE_float('test_epsilon', 0.01, """epsilon greedy""")
tf.app.flags.DEFINE_float('sample_weight', 0.7, """sampling weight""")
tf.app.flags.DEFINE_integer('trainbatch', 64, """training batch size""")
tf.app.flags.DEFINE_integer('samplesize', 128, """samples""")
tf.app.flags.DEFINE_integer('replay_size', 1000000, """replay buffer size""")
tf.app.flags.DEFINE_integer('train_iterations', 1, """training iterations""")
tf.app.flags.DEFINE_integer('observations', 5000, """initial observations""")
tf.app.flags.DEFINE_integer('save_interval', 5000, """intervals to save model""")
tf.app.flags.DEFINE_float('gamma', 0.7, """gamma""")
tf.app.flags.DEFINE_integer('copy_network_interval', 2000, """intervals to copy network from qnet to targetnet""")

class Network:
    def __init__(self, size, scope, session, readonly = True):
        self.size = size
        self.scope = scope
        self.readonly = readonly
        tf.contrib.framework.get_or_create_global_step()
        with tf.variable_scope(scope):
            self.init_network()
        self.session = session
        self.saver = tf.train.Saver(self.savable_variables())
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        self.path = os.path.join(FLAGS.model_dir, self.scope)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.model_location = os.path.join(self.path, "model")

    def savable_variables(self):
        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        if not self.readonly:
            vs.append(tf.contrib.framework.get_global_step())
        return vs

    def save(self):
        assert(self.saver)
        steps = self.session.run(tf.contrib.framework.get_global_step())
        checkpoint = tf.train.latest_checkpoint(self.path)
        self.saver.save(self.session, self.model_location, steps)
        print("SAVING PARAMETERS STEP = {}".format(steps))

    def restore(self):
        assert(self.saver)
        latest_checkpoint = tf.train.latest_checkpoint(self.path)
        if latest_checkpoint:
            self.saver.restore(self.session, latest_checkpoint)
            steps = self.session.run(tf.contrib.framework.get_global_step())
            print("RESTORE PARAMETERS {} STEP = {}".format(latest_checkpoint, steps))

    def init_network(self):
        flatsize = self.size ** 2
        self.input = tf.placeholder(dtype = tf.float32,
                shape = [None, self.size, self.size, 3],
                name = "input")
        self.y = tf.placeholder(shape = [None], dtype = tf.float32,
                name = "y")
        self.actions = tf.placeholder(shape=[None, 2], dtype=tf.int32,
                name = "actions")
        self.actions_flat = tf.reshape( \
                tf.slice(self.actions, [0, 0], [-1, 1]) \
                * self.size + tf.slice(self.actions, [0, 1], [-1, 1]), [-1])

        conv1 = tf.contrib.layers.conv2d(self.input, 8, 5, 1,
                activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 3, 1,
                activation_fn=tf.nn.relu)
        flat= tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(flat, 8 * flatsize)
        self.predictions=tf.contrib.layers.fully_connected(fc1, flatsize)
        samples = tf.shape(self.input)[0]
        masks = tf.reshape(tf.slice(self.input, [0, 0, 0, 2], [-1, -1, -1, 1]),
                [samples, -1])
        rev_masks = 1.0 - masks
        self.legal_moves = self.predictions * rev_masks  + \
                tf.transpose(tf.transpose(masks) *         \
                (tf.reduce_min(self.predictions, reduction_indices=[1]) - 1.0))

        # [0, output, 2*output, ..., (samples-1)*output] + actions
        self.index = tf.range(samples) * tf.shape(self.predictions)[1] + \
                self.actions_flat
        self.pred = tf.gather(tf.reshape(self.predictions, [-1]), self.index)
        self.losses = tf.squared_difference(self.y, self.pred)
        self.loss = tf.reduce_mean(self.losses)
        self.optimizer=tf.train.RMSPropOptimizer(FLAGS.learn_rate, 0.99)
        self.train = self.optimizer.minimize(self.loss,
                global_step=tf.contrib.framework.get_global_step())

class CopyNetwork:
    def __init__(self, src, dst):
        w_src = [s for s in tf.trainable_variables()
                if s.name.startswith(src.scope)]
        w_dst = [s for s in tf.trainable_variables()
                if s.name.startswith(dst.scope)]
        self.copy_ops = []
        for s, d in zip(w_src, w_dst):
            self.copy_ops.append(d.assign(s))

    def copy(self, session):
        session.run(self.copy_ops)

class DqnAgent(Agent):
    state_self     = 0
    state_opponent = 1
    state_mask     = 2
    def __init__(self, size, session):
        super().__init__(size, session)
        self.scope     = None
        self.test_mode = False
        self.q_network = None
        self.network_restored = False
        self.epsilon = 0.0
    def clear(self):
        super().clear()
        self.board = np.zeros((self.size, self.size), dtype = np.int32), \
                     np.zeros((self.size, self.size), dtype = np.int32), \
                     np.zeros((self.size, self.size), dtype = np.int32)
        if self.network_restored == False:
            assert(self.q_network != None)
            self.q_network.restore()
            self.network_restored = True
    def init_network(self, path, ro = True):
        self.q_network = Network(self.size, self.scope + '/q', self.session, ro)
    def update_state_(self, position, mover):
        mask = self.board[2]
        mover_board = self.board[mover]
        assert(mask[position] == 0.0 and mover_board[position] == 0.0)
        mask[position]  = 1.0
        mover_board[position] = 1.0
    def select(self, input, network):
        stacked = np.stack(input, axis = 0)
        pred, out = network.session.run([network.predictions,
                    network.legal_moves],
                    feed_dict = { network.input : stacked } )
        m = np.argmax(out, axis = 1)
        return m, out
    def self_move(self):
        _, _, mask = self.board
        if random.uniform(0, 1) < self.epsilon:
            move = super().random_policy(mask)
        else:
            inputs = [np.stack(self.board, axis = -1)]
            m, _ = self.select(inputs, self.q_network)
            move = m[0] // self.size, m[0] % self.size
        self.update_state_(move, DqnAgent.state_self)
        return move
    def opponent_move(self, position):
        self.update_state_(position, DqnAgent.state_opponent)

class DqntrainAgent(DqnAgent):
    def __init__(self, size, session):
        super().__init__(size, session)
        self.game_queue = []
        self.replay = deque()
        self.scope = FLAGS.train_generation
        self.nr_games = 0
        self.losses = []
        self.epsilon = FLAGS.train_epsilon
        if not self.test_mode:
            self.init_network()
    def clear(self):
        super().clear()
        self.opponent_mv = None
        self.self_mv = None
        self.orig_board = None
    def init_network(self):
        super().init_network(self.scope, ro = False)
        self.target_network = Network(self.size, self.scope + '/target', self.session)
        self.copy_network = CopyNetwork(self.q_network, self.target_network)

    def name(self):
        return "@"
    def append_replay_buf(self, orig, self_mv, opponent_mv, new, reward, endgame):
        self.game_queue.append((orig, self_mv, opponent_mv, new, reward, endgame))
    def self_move(self):
        a, b, c = self.board
        new_board = a.copy(), b.copy(), c.copy()
        if self.orig_board:
            self.append_replay_buf(self.orig_board, self.self_mv,
                    self.opponent_mv, new_board, 0.0, False)
        self.orig_board = new_board
        self.self_mv = super().self_move()
        return self.self_mv
    def opponent_move(self, move):
        super().opponent_move(move)
        self.opponent_mv = move
    def train_step(self):
        samples = random.sample(self.replay, FLAGS.samplesize)
        weights = list(map(lambda s: s[6], samples))
        tot = sum(weights)
        prob = list(map(lambda s: s / tot, weights))
        index = np.random.choice(range(FLAGS.samplesize),
                size = FLAGS.trainbatch, p = prob)
        mini_batch = [samples[i] for i in index]

        inputs = []
        ends = []
        orig = []
        actions = []
        rewards = []
        for sample in mini_batch:
            orig_board, self_mv, opponent_mv, new_board, reward, end, _ = sample
            inputs.append(np.stack(new_board, axis = -1))
            ends.append(1 - end)
            orig.append(np.stack(orig_board, axis = -1))
            actions.append(self_mv)
            rewards.append(reward)
        move, _ = self.select(inputs, self.q_network)
        _, outtgt = self.select(inputs, self.target_network)
        best = np.array(ends) * outtgt[range(len(move)), move]
        outtgt = rewards + best * FLAGS.gamma

        step_var = tf.contrib.framework.get_global_step()
        _, loss, steps = self.q_network.session.run(
                [self.q_network.train, self.q_network.loss, step_var],
                feed_dict = { self.q_network.input : orig,
                              self.q_network.y : outtgt,
                              self.q_network.actions : actions } )
        if steps > 0 and steps % FLAGS.save_interval == 0:
            self.q_network.save()
        self.losses.append(loss)

    def finish(self, reward):
        a, b, c = self.board
        new_board = a.copy(), b.copy(), c.copy()
        if self.orig_board:
            self.append_replay_buf(self.orig_board, self.self_mv,
                self.opponent_mv, new_board, reward, True)
        weight = 1.0
        while len(self.game_queue) > 0:
            a,b,c,d,e,f = self.game_queue.pop()
            self.replay.append((a, b, c, d, e, f, weight))
            weight *= FLAGS.sample_weight
            if len(self.replay) > FLAGS.replay_size:
                self.replay.popleft()
        if self.nr_games > FLAGS.observations:
            for _ in range(FLAGS.train_iterations):
                self.train_step()
        if self.nr_games % FLAGS.copy_network_interval == 0:
            self.copy_network.copy(self.session)
        self.nr_games += 1
    def info(self):
        if not self.losses:
            return
        print(" loss = {:.3f} ".format(sum(self.losses) / len(self.losses)), end="")
        self.losses = []

class DqntestAgentOne(DqnAgent):
    def __init__(self, size, session, scope):
        super().__init__(size, session)
        self.scope = scope
        self.epsilon = FLAGS.test_epsilon
        self.init_network(self.scope, ro = True)
    def name(self):
        return "#"

class DqntestAgent(Agent):
    def __init__(self, size, session):
        super().__init__(size, session)
        self.scopes = FLAGS.test_generation.split(",")
        self.agents = []
        for scope in self.scopes:
            self.agents.append(DqntestAgentOne(size, session, scope))
    def name(self):
        return "#"
    def opponent_move(self, position):
        self.active.opponent_move(position)
    def self_move(self):
        return self.active.self_move()
    def finish(self, result):
        active = self.active
        return self.active.finish(result)
    def clear(self):
        super().clear()
        self.active = random.sample(self.agents, 1)[0]
        self.active.clear()
    def info(self):
        pass

class CloneNetworks:
    def __init__(self, size, session):
        self.session = session
        self.src = Network(size, FLAGS.copy_from + '/q', session, readonly= True)
        self.dst = Network(size, FLAGS.copy_to + '/q', session, readonly = False)
        self.copynetwork = CopyNetwork(self.src, self.dst)
    def copy(self):
        self.src.restore()
        self.copynetwork.copy(self.session)
        self.dst.save()
