from random import Random
from typing import Any, Dict, Hashable, List, Optional, Tuple

import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

from environment import ActionId, State

from .rl import Policy


def _add_dense_layer(x, num_input, num_output, add_relu):
    W = tf.get_variable("w", [num_input, num_output], initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float64), dtype=tf.float64)
    b = tf.get_variable("b", [num_output], initializer=tf.constant_initializer(0.1, dtype=tf.float64), dtype=tf.float64)
    x = tf.matmul(x, W) + b
    if add_relu:
        x = tf.nn.relu(x)
    return x, [W, b]


def create_net(x, num_input, num_output, num_neurons, num_hidden_layers):
    weights = []
    for i in range(num_hidden_layers):
        with tf.variable_scope("Layer_" + str(i)):
            layer_in  = num_input if i == 0 else num_neurons
            layer_out = num_output if i == num_hidden_layers-1 else num_neurons
            add_relu  = i < num_hidden_layers-1
            x, w = _add_dense_layer(x, layer_in, layer_out, add_relu)
            weights.extend(w)
    return x, weights


def create_target_updates(weights, target_weights):
    init_updates = []
    for i in range(len(weights)):
        init_updates.append(tf.assign(target_weights[i], weights[i]))
    return init_updates


class NN:
    def __init__(self, sess: Any, alpha: float, gamma: float, default_q: float,
                 num_features: int, num_actions: int, num_neurons: int,
                 num_hidden_layers: int, name: str):
        self.sess = sess
        self.name = name

        # Inputs to the network
        self.s1 = tf.placeholder(tf.float64, [None, num_features])
        self.a = tf.placeholder(tf.int32)
        self.r = tf.placeholder(tf.float64)
        self.s2 = tf.placeholder(tf.float64, [None, num_features])
        self.end = tf.placeholder(tf.float64)
        self.filter_mask = tf.placeholder(tf.float64, [None, num_actions])

        # Creating target and current networks
        with tf.variable_scope(name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            with tf.variable_scope("q_network") as scope:
                q_values, q_values_weights = create_net(self.s1, num_features,
                                                        num_actions,
                                                        num_neurons,
                                                        num_hidden_layers)
                scope.reuse_variables()
                q2_values, _ = create_net(self.s2, num_features, num_actions,
                                          num_neurons, num_hidden_layers)
            with tf.variable_scope("q_target"):
                q_target, q_target_weights = create_net(self.s2, num_features,
                                                        num_actions,
                                                        num_neurons,
                                                        num_hidden_layers)
            self.update_target = create_target_updates(q_values_weights,
                                                       q_target_weights)

            # Q_values -> get optimal actions
            # self.best_action = tf.argmax(q_values, 1)
            self.best_action = tf.argmax(q_values -
                    10000000 * (1.0 - self.filter_mask), 1)

            # Optimizing with respect to q_target
            action_mask = tf.one_hot(indices=self.a, depth=num_actions,
                                     dtype=tf.float64)
            q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)

            best_action_mask = tf.one_hot(indices=tf.argmax(q2_values, 1),
                                          depth=num_actions, dtype=tf.float64)
            q_max = tf.reduce_sum(tf.multiply(q_target, best_action_mask), 1)

            # Computing td-error and loss function
            q_max = q_max * (1.0 - self.end) # dead ends must have q_max equal to zero
            q_target_value = self.r + gamma * q_max
            q_target_value = tf.stop_gradient(q_target_value)
            loss = 0.5 * tf.reduce_sum(tf.square(q_current - q_target_value))

            # Defining the optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
            self.train = optimizer.minimize(loss=loss)

    def initialize(self):
        self.sess.run(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=self.name)))
        self.sess.run(self.update_target)  # copying weights to target net


class ReplayBuffer:
    data: List[Any]
    limit: int
    pointer: int
    rng: Random

    def __init__(self, buffer_size: int, rng: Random):
        self.data = []
        self.limit = buffer_size
        self.pointer = 0
        self.rng = rng

    def add(self, element: Any):
        if self.pointer == len(self.data):
            self.data.append(element)
        else:
            self.data[self.pointer] = element
        self.pointer += 1
        if self.pointer > self.limit:
            self.pointer = 0

    def sample(self, count: int) -> List[Any]:
        return [self.rng.choice(self.data) for _ in range(count)]


class DQN(Policy):
    alpha: float
    batch_size: int
    buffer_size: int
    default_q: Tuple[float, bool]
    frequency: int
    gamma: float
    name: str
    num_actions: int
    num_features: int
    num_neurons: int
    num_hidden_layers: int
    rng: Random
    sess: Any
    start: int
    step: int

    def __init__(self, alpha: float, gamma: float, default_q: float,
                 num_features: int, num_actions: int, num_neurons: int,
                 num_hidden_layers: int, batch_size: int, buffer_size: int,
                 rng: Random, sess: Any, start: int, frequency: int,
                 name: str):
        self.alpha = alpha
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.default_q = default_q, False
        self.frequency = frequency
        self.gamma = gamma
        self.name = name
        self.num_actions = num_actions
        self.num_features = num_features
        self.num_neurons = num_neurons
        self.num_hidden_layers = num_hidden_layers
        self.rng = rng
        self.sess = sess
        self.start = start

        self.network = NN(self.sess, self.alpha, self.gamma, self.default_q[0],
                          self.num_features, self.num_actions,
                          self.num_neurons, self.num_hidden_layers, self.name)
        self.clear()

    def clear(self):
        self.network.initialize()
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.rng)
        self.step = 0

    def encode(self, state: State) -> Any:
        s: List[float] = []
        raise NotImplementedError
        # for i in state.uid:
        #     if isinstance(i, tuple):
        #         for j in i:
        #             s.append(float(j))
        #     else:
        #         s.append(float(i))
        return np.array(s, dtype=np.float64)

    def get_best_action(self, state: State,
                        restrict: Optional[List[int]] = None) -> ActionId:
        s = self.encode(state).reshape(1, self.num_features)
        if restrict is None:
            mask = np.ones((1, self.num_actions), dtype=np.float)                
        else:
            mask = np.zeros((1, self.num_actions), dtype=np.float64)
            for i in restrict:
                mask[0][i] = 1.0
        a = self.sess.run(self.network.best_action,
                          {self.network.s1: s, self.network.filter_mask: mask})
        return a[0]

    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None) -> ActionId:
        return self.get_best_action(state, restrict)

    def learn(self):
        self.step += 1
        if self.step < self.start:
            return

        data = self.replay_buffer.sample(self.batch_size)

        S1, A, S2, R, End = [], [], [], [], []
        for s1, a, s2, r, end in data:
            S1.append(self.encode(s1))
            A.append(a)
            S2.append(self.encode(s2))
            R.append(r)
            End.append(end)

        self.sess.run(self.network.train,
                      {self.network.s1: np.array(S1),
                       self.network.a: np.array(A),
                       self.network.r: np.array(R),
                       self.network.s2: np.array(S2),
                       self.network.end: np.array(End)})

        if self.step % self.frequency == 0:
            self.sess.run(self.network.update_target)

    def update(self, s0: State, a: ActionId, s1: State, r: float, end: bool):
        self.replay_buffer.add((s0, a, s1, r, end))
        self.learn()

    def reset(self, evaluation: bool):
        self.evaluation = evaluation

    def report(self) -> str:
        return ""


class EpsilonDQN(DQN):
    epsilon: float

    def __init__(self, alpha: float, gamma: float, default_q: float,
                 num_features: int, num_actions: int, num_neurons: int,
                 num_hidden_layers: int, batch_size: int, buffer_size: int,
                 rng: Random, sess: Any, start: int, frequency: int, name: str,
                 epsilon: float):
        self.epsilon = epsilon
        super().__init__(alpha, gamma, default_q, num_features, num_actions,
                         num_neurons, num_hidden_layers, batch_size,
                         buffer_size, rng, sess, start, frequency, name)

    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None) -> ActionId:
        if self.rng.random() < self.epsilon:
            if restrict is None:
                restrict = list(range(self.num_actions))
            return self.rng.choice(restrict)
        else:
            return self.get_best_action(state, restrict)
