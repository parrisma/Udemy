from typing import Dict
import numpy as np
import gym
from src.cartpole.agent import Agent
from src.cartpole.agent_factory import AgentFactory


class RandomAgent(Agent):
    """
    Agent where choice is random over the action space.
    """

    def __init__(self,
                 _: gym.Env):
        return

    def chose_action(self,
                     _: np.ndarray) -> int:
        """
        Action is random and not dependant on current state
        """
        return np.random.choice([0, 1], 1)[0]

    def reset(self):
        return

    def reward(self,
               state: np.ndarray,
               action: int,
               reward: float) -> None:
        return

    def done(self) -> None:
        return

    def final(self) -> None:
        return

    def debug(self) -> Dict:
        return dict()

class RandomAgentFactory(AgentFactory):
    @staticmethod
    def new(env: gym.Env) -> Agent:
        return RandomAgent(env)
