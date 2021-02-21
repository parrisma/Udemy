from typing import List, Dict
import random
import numpy as np
import gym
from src.cartpole.agent import Agent
from src.cartpole.agent_factory import AgentFactory
import tensorflow as tf
import collections


class DQNModelAgent(Agent):
    """
    Agent using Q Learning with continuous feature space discretized into 10 bins per feature
    """
    _q_learning_rate: float
    _num_actions: int
    _actions: List[int]
    _num_states: int
    _q: np.ndarray
    _epsilon: float
    _gamma: float
    _global_iter: int
    _main_dqn = tf.keras.Model
    _main_training_count: int
    _target_dqn = tf.keras.Model
    _buffer: collections.deque

    def __init__(self,
                 env: gym.Env):
        self._q_learning_rate = 1e-2
        self._actions = list(range(0, env.action_space.n))
        self._num_actions = len(self._actions)
        self._num_states = 1
        self._epsilon = 1.0
        self._gamma = 0.999
        self._global_iter = 1  # avoid div by zero
        self._main_dqn = self._create_model()
        self._main_training_count = 1
        self._target_dqn = self._create_model()
        self._target_upd = 0
        self._buffer = collections.deque(maxlen=2500)  # s,a,r,s'
        self._act_hist = collections.deque(maxlen=250)
        self.save_res("res_left_m.csv")
        self.save_res("res_left_t.csv")
        self.save_res("res_right_m.csv")
        self.save_res("res_right_t.csv")

        return

    def _create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(input_shape=(self._num_states,), units=1, name='Input'),
            tf.keras.layers.Dense(200, activation=tf.nn.relu, name='dense1'),
            tf.keras.layers.Dropout(rate=0.5, name="Dropout"),
            tf.keras.layers.Dense(200, activation=tf.nn.relu, name='dense2'),
            tf.keras.layers.Dense(self._num_actions, name='output')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
            loss=tf.keras.losses.mean_squared_error
        )
        return model

    def init(self,
             env) -> None:
        """
        Called once at the start of a session
        """
        return

    def _predict(self,
                 state: np.ndarray) -> np.ndarray:
        """
        Predict the Q Values for each action and select the greedy action
        :param state: The current environment state
        :return: The greedy action
        """
        return self._target_dqn.predict(x=state)

    def chose_action(self, state: np.ndarray) -> int:
        """
        Take random action with probability (current) epsilon or greedy action based on current
        Q Values by action for given state
        :param state: The current environment state
        :return: The action to take in the environment.
        """
        if True:  # np.random.random() < self._epsilon:
            action = np.random.choice(self._actions, 1)[0]
        else:
            state = state[2].reshape((1))
            p = self._predict(np.atleast_2d(state))
            action = np.argmax(p[0])
            self._act_hist.append(action)
        if self._main_training_count % 10 == 0 and len(self._act_hist) > 50:
            last_50 = list(self._act_hist)[:-50]
            print("{} : {}".format(np.average(np.array(last_50)), ",".join([str(a) for a in last_50])))
        return action

    def _train(self,
               samples: List) -> None:
        """
        Select a random batch of updates from the buffer and train the main_dqn. Then every 100
        times main_dqn is trained train the target network.
        """
        prev_state = np.empty((len(samples), self._num_states))
        curr_state = np.empty((len(samples), self._num_states))
        actions = np.empty((len(samples), self._num_states))
        rewards = np.empty((len(samples), self._num_states))
        y = np.empty((len(samples), self._num_actions))

        i = 0
        for s in samples:
            prev_state[i] = s[0]
            actions[i] = s[1]
            rewards[i] = s[2]
            curr_state[i] = s[3]
            i += 1

        prev_state_preds = self._predict(prev_state)
        curr_state_preds = self._predict(curr_state)

        i = 0
        for _, action, reward, _ in samples:
            if reward < 0:
                i = i
            G = reward + self._gamma * np.max(curr_state_preds[i])
            yi = prev_state_preds[i]
            yi[action] += self._q_learning_rate * (G - np.max(yi))  # yi[action]
            y[i] = yi
            i += 1

        self._main_training_count += 1
        if self._main_training_count % 25 == 0:
            self._main_dqn.fit(prev_state, y, batch_size=50, epochs=25, verbose=0, validation_split=0.2)
            res = np.array([(self._main_dqn.predict([x])).reshape(2).tolist() for x in
                            (np.array(range(-140, 150, 10)) * (.18 / 100))])
            res_l = res[:, 0]
            res_r = res[:, 1]
            self.save_res("res_left_m.csv", res_l)
            self.save_res("res_right_m.csv", res_r)
        else:
            self._main_dqn.fit(prev_state, y, batch_size=50, epochs=25, verbose=0)

        if self._main_training_count % 100 == 0:
            for i in range(len(self._target_dqn.trainable_weights)):
                self._target_dqn.trainable_weights[i].assign(self._main_dqn.trainable_weights[i])
            self._target_upd += 1
            print("Target Update {}".format(self._target_upd))
            res = np.array([(self._target_dqn.predict([x])).reshape(2).tolist() for x in
                            (np.array(range(-140, 150, 10)) * (.18 / 100))])
            res_l = res[:, 0]
            res_r = res[:, 1]
            self.save_res("res_left_t.csv", res_l)
            self.save_res("res_right_t.csv", res_r)
        return

    def save_res(self,
                 file_name: str,
                 res: np.ndarray = None) -> None:
        if res is None:
            f = open(file_name, "w")
            f.close()
        else:
            f = open(file_name, "a")
            f.write(",".join([str(x) for x in res.tolist()]))
            f.write("\n")
            f.flush()
            f.close()
        return

    def reward(self,
               state: np.ndarray,
               prev_state: np.ndarray,
               action: int,
               reward: float) -> None:
        """
        Update Q with respect to given reward for State/Action pair
        :param state: The state the action was taken in
        :param prev_state: The state previous to the given current state
        :param action: The action taken
        :param reward: The reward for the State/Action
        """
        state = state[2].reshape((1))
        prev_state = prev_state[2].reshape((1))

        self._buffer.append([prev_state,
                             action,
                             reward,
                             state])
        if len(self._buffer) >= 500:
            self._train(random.sample(self._buffer, 500))
        return

    def done(self) -> None:
        self._epsilon = 1.0
        if self._target_upd > 7:
            self._global_iter += 1
            self._epsilon = 1.0 / np.sqrt(self._target_upd + 1)
        return

    def reset(self):
        """
        No episode specific reset
        """
        return

    def final(self) -> None:
        """
        :return:
        """
        return

    def debug(self) -> Dict:
        return {"epsilon": self._epsilon}


class DQNModelAgentFactory(AgentFactory):
    @staticmethod
    def new(env: gym.Env) -> Agent:
        return DQNModelAgent(env)
