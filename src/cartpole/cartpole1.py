import abc
import sys
from typing import List, Tuple, Any
import gym
import numpy as np

"""
Udemy.
# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python

Personal implementation of simple linear model & random search for weights. (no learning)
"""


class Util:
    @staticmethod
    def stats(history: np.ndarray) -> Tuple:
        return np.min(history), np.max(history), np.mean(history), np.std(history)


class Agent(metaclass=abc.ABCMeta):
    """
    Interface spec for a CartPole Agent
    """

    def reset(self, **kwargs):
        """
        Reset the agent before start of a new episode
        :return:
        """
        raise NotImplementedError()

    def chose_action(self,
                     state: np.ndarray) -> int:
        """
        Chose an action - assumes
        :return: The action from the action space [0, 1]
        """
        raise NotImplementedError()

    def done(self) -> None:
        """
        Called when episode is complete
        """
        raise NotImplementedError()


class RandomAgent(Agent):
    """
    Action choice is random over the action space.
    """

    def reset(self, **kwargs):
        return

    def chose_action(self,
                     _: np.ndarray) -> int:
        """
        Action is random and not dependant on current state
        """
        return np.random.choice([0, 1], 1)[0]

    def done(self):
        return


class LinearModelAgent(Agent):
    _weights: np.ndarray
    _best_weights: np.ndarray
    _episode_len: int
    _best_len: int
    _episode_count: int
    _reset_every: int
    _len_hist: List

    def __init__(self):
        self._weights = None  # noqa
        self._episode_len = 0
        self._best_len = 0
        self._best_weights = None  # noqa
        self._episode_count = 0
        self._reset_every = 100  # check to see if average len is better then best so far and if so update weights
        self._len_hist = list()
        self.reset(state=None)
        self._new_weights()

    def reset(self, **kwargs):
        self._episode_len = 0
        return

    def _new_weights(self) -> None:
        """
        Update model weights for linear model.
        """
        self._weights = np.random.random(4) * 2 - 1
        return

    def chose_action(self,
                     state: np.ndarray) -> int:
        """
        Apply simple linear model to determine action to take.
        :param state: The current state to apply weights to
        """
        self._episode_len += 1
        return 1 if state.dot(self._weights) > 0 else 0

    def done(self) -> None:
        """
        Update the history of episode lengths. Then every 100 episodes check to see if the average
        game length is better than the previous best; if it is save the current (better) weights as
        best weights.
        """
        self._len_hist.append(self._episode_len)
        self._episode_count += 1
        if self._episode_count % self._reset_every == 0:
            ave_len = int(np.mean(np.asarray(self._len_hist)))
            if self._best_len < ave_len:
                self._best_weights = self._weights
                print(
                    "episode : [{}] Found better weights given mean episode len from [{}] up to [{}]".format(
                        self._episode_count, self._best_len, ave_len))
                self._best_len = ave_len
            self._len_hist = list()
            self._new_weights()
        return


class CartPole1:

    def __init__(self,
                 agent: Agent):
        self._env = gym.make('CartPole-v0')
        self._env.reset()
        self._agent = agent
        return

    def run(self,
            num_episodes: int) -> List[int]:
        """
        Execute the given number of episodes
        :param num_episodes: The number of episodes to play
        :return: The episode length history
        """
        history = list()
        for e in range(0, num_episodes):
            done = False
            n = 0
            state = self._env.reset()
            self._agent.reset()
            while not done and n < 201:  # 200 is max episodes allowed by CartPole
                action = self._agent.chose_action(state)
                state, reward, done, _ = self._env.step(action)
                n += 1
            history.append(n)
            self._agent.done()
            if e % 1000 == 0:
                print("episode : [{}]".format(e))
        return history


if __name__ == "__main__":
    if 'random' in sys.argv:
        hist = CartPole1(RandomAgent()).run(100)
    else:
        hist = CartPole1(LinearModelAgent()).run(5000)

    print("Min [{:12.6f}] Max [{:12.6f}] Ave:[{:12.6f}] StdDev:[{:12.6f}]".format(*Util.stats(np.asarray(hist))))
