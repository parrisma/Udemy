from typing import Dict
import abc
import numpy as np


class Agent(metaclass=abc.ABCMeta):
    """
    Interface spec for a CartPole Agent
    """

    @abc.abstractmethod
    def reset(self):
        """
        Reset the agent before start of a new episode
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def chose_action(self,
                     state: np.ndarray) -> int:
        """
        Chose an action - assumes
        :return: The action from the action space [0, 1]
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reward(self,
               state: np.ndarray,
               prev_state: np.ndarray,
               action: int,
               reward: float) -> None:
        """
        Update agent based on reward for playing action in given state
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def done(self) -> None:
        """
        Called when episode is complete
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def final(self) -> None:
        """
        Called after all episodes are complete
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def debug(self) -> Dict:
        """
        Return a dictionary of model specific debug data
        :return:
        """
        raise NotImplementedError()
