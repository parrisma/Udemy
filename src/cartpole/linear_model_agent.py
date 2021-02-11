from typing import List, Dict
import numpy as np
import gym
from src.cartpole.agent import Agent
from src.cartpole.agent_factory import AgentFactory


class LinearModelAgent(Agent):
    """
    Agent to execute random search for a linear model
    """
    _weights: np.ndarray
    _best_weights: np.ndarray
    _episode_len: int
    _best_len: int
    _episode_count: int
    _reset_every: int
    _len_hist: List
    _states: List[np.ndarray]

    def __init__(self,
                 _: gym.Env):
        self._weights = None  # noqa
        self._episode_len = 0
        self._best_len = 0
        self._best_weights = None  # noqa
        self._episode_count = 0
        self._reset_every = 100  # check to see if average len is better then best so far and if so update weights
        self._len_hist = list()
        self.reset()
        self._new_weights()
        self._states = list()

    def init(self, env):
        return

    def reset(self):
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
        self._states.append(state)
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

    def final(self) -> None:
        """
        Report the state space ranges and proposed bins for RBS experiment.
        """
        states = np.asarray(self._states)

        print("State bounds to determine bins for finite state Q learning model")
        for state_name, state_values in [['Position     ', states[:, 0]],
                                         ['Velocity     ', states[:, 1]],
                                         ['Pole Angle   ', states[:, 2]],
                                         ['Pole Velocity', states[:, 3]]
                                         ]:
            print("{}: Min [{:12.6f}] Max [{:12.6f}] Ave:[{:12.6f}] StdDev:[{:12.6f}]".format(state_name,
                                                                                              np.min(state_values),
                                                                                              np.max(state_values),
                                                                                              np.mean(state_values),
                                                                                              np.std(state_values)))
        return

    def reward(self,
               state: np.ndarray,
               action: int,
               reward: float) -> None:
        return

    def debug(self) -> Dict:
        return dict()

class LinearModelAgentFactory(AgentFactory):
    @staticmethod
    def new(env: gym.Env) -> Agent:
        return LinearModelAgent(env)
