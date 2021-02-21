from typing import List, Dict, Tuple
import os
import numpy as np
import gym
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mgimg
from src.cartpole.agent import Agent
from src.cartpole.agent_factory import AgentFactory


class YFeatureTransformer:
    def __init__(self):
        # Simple linear bins, but we could use stddev to determine bin boundaries to
        # smooth distribution of samples across bins.
        # bounds indicated by the LinearModelAgent at the end of its run.
        # self._pos_bins = np.linspace(-2.4, 2.4, self.num_bins - 1)
        # self._vel_bins = np.linspace(-3.5, 3.5, self.num_bins - 1)
        self._pole_angle_bins = np.linspace(-0.25, 0.25, self.num_bins - 1)
        self._pole_vel_bins = np.linspace(-3.5, 3.5, self.num_bins - 1)
        return

    @property
    def num_bins(self) -> int:
        return 10

    @staticmethod
    def build_state(features) -> int:
        return int("".join(map(lambda feature: str(int(feature)), features)))

    @staticmethod
    def to_bin(val: float, bins):
        return np.digitize(x=[val], bins=bins)[0]

    def transform(self, state) -> int:
        """
        Convert state to single value
        :param state:
        :return:
        """
        pos, vel, pole_angle, pole_vel = state
        return self.build_state([
            self.to_bin(pole_vel, self._pole_vel_bins),
            self.to_bin(pole_angle, self._pole_angle_bins)
        ])


class YQModelAgent(Agent):
    """
    Agent using Q Learning with continuous feature space discretized into 10 bins per feature
    """
    _feature_transformer: YFeatureTransformer
    _learning_rate: float
    _num_actions: int
    _actions: List[int]
    _num_states: int
    _q: np.ndarray
    _epsilon: float
    _gamma: float
    _global_iter: int
    _master_iter: int
    upd = dict()

    def __init__(self,
                 env: gym.Env):
        self._feature_transformer = YFeatureTransformer()
        self._learning_rate = 10e-3
        self._actions = list(range(0, env.action_space.n))
        self._num_actions = len(self._actions)
        self._num_states = self._feature_transformer.num_bins ** 2  # Two features
        self._q = np.random.uniform(low=-1, high=1, size=(self._num_states, self._num_actions))
        self._epsilon = 1.0
        self._gamma = 0.9
        self._global_iter = 1  # avoid div by zero
        self._master_iter = 1
        self._fig_num = 1
        self._fig, self._axs = plt.subplots(2, 2)
        self._fig.tight_layout()
        plt.ion()
        return

    def init(self,
             env) -> None:
        """
        Called once at the start of a session
        """
        return

    def _q2mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the q values as a mesh for plotting.
        :return: q values as mash Tuple(bins, left action q vals, right action q values)
        """
        qmeshl = np.zeros((self._feature_transformer.num_bins, self._feature_transformer.num_bins))
        qmeshr = np.zeros((self._feature_transformer.num_bins, self._feature_transformer.num_bins))
        bins = range(self._feature_transformer.num_bins)
        for i in bins:  # pole_vel
            for j in bins:  # pole_angle
                qmeshl[i, j] = self._q[int("{}{}".format(i, j))][0]
                qmeshr[i, j] = self._q[int("{}{}".format(i, j))][1]
        return np.array(bins), qmeshl, qmeshr

    def _predict(self,
                 state: np.ndarray) -> int:
        """
        Predict the Q Values for each action and select the greedy action
        :param state: The current environment state
        :return: The greedy action
        """
        discrete_state = self._feature_transformer.transform(state)
        return self._q[discrete_state]

    def chose_action(self, state: np.ndarray) -> int:
        """
        Take random action with probability (current) epsilon or greedy action based on current
        Q Values by action for given state
        :param state: The current environment state
        :return: The action to take in the environment.
        """
        if np.random.random() < self._epsilon:
            action = np.random.choice(self._actions, 1)[0]
        else:
            p = self._predict(state)
            action = np.argmax(p)
        return action

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
        G = reward + self._gamma * np.max(self._predict(state))
        discrete_state = self._feature_transformer.transform(prev_state)
        # self._q[discrete_state, action] += self._learning_rate * (G - self._q[discrete_state, action])
        self._q[discrete_state, action] += self._learning_rate * (G - np.max(self._q[discrete_state]))
        if self._master_iter == 1 or self._master_iter % 10000 == 0:
            self._visualise(*self._q2mesh())
        self._master_iter += 1
        return

    def done(self) -> None:
        self._global_iter += 1
        self._epsilon = 1.0 / np.sqrt(self._global_iter)
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

    def _visualise(self,
                   ax: np.ndarray,
                   ql: np.ndarray,
                   qr: np.ndarray) -> None:
        """
        Render the visual plot of the current state
        :param ax: The axis bins (all features have same bins so only need once)
        :param ql: The Q Values by bin for the left action
        :param qr: The Q Values by bin for the right action
        """
        self._fig, self._axs = plt.subplots(2, 2)
        self._fig.tight_layout()
        s1 = self._axs[0, 0].contourf(ax, ax, ql, 20, cmap=cm.coolwarm, antialiased=False)
        s2 = self._axs[0, 1].contourf(ax, ax, qr, 20, cmap=cm.coolwarm, antialiased=False)
        s3 = self._axs[1, 0].contourf(ax, ax, ql - qr, 20, cmap=cm.viridis, antialiased=False)
        lr = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                if ql[i, j] > qr[i, j]:  # max q value is the greedy action to take.
                    lr[i, j] = 1
                else:
                    lr[i, j] = -1
        s4 = self._axs[1, 1].pcolormesh(ax, ax, lr, vmin=np.min(lr), vmax=np.max(lr), cmap=cm.viridis, shading='auto')
        self._fig.colorbar(s1, ax=self._axs[0, 0])
        self._fig.colorbar(s2, ax=self._axs[0, 1])
        self._fig.colorbar(s3, ax=self._axs[1, 0])
        self._fig.colorbar(s4, ax=self._axs[1, 1])
        self._axs[0, 0].set_title("Q Left")
        self._axs[0, 0].set_ylabel("Pole Angle")
        self._axs[0, 1].set_title("Q Right")
        self._axs[1, 0].set_title("Q Diff")
        self._axs[1, 0].set_ylabel("Pole Angle")
        self._axs[1, 0].set_xlabel("Pole Velocity")
        self._axs[1, 1].set_title("Left / Right")
        self._axs[1, 1].set_xlabel("Pole Velocity")
        plt.savefig("./images/cartpole_{}.png".format(self._fig_num))
        self._fig_num += 1
        plt.show()
        return

    def _animate(self) -> None:
        """
        Convert the saved plots into a animation (movie)
        :return:
        """
        fig_num = 1
        while os.path.exists("./images/cartpole_{}.png".format(self._fig_num)):

            fig_num += 1
        return


class YQModelAgentFactory(AgentFactory):
    @staticmethod
    def new(env: gym.Env) -> Agent:
        return YQModelAgent(env)
