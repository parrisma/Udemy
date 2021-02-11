import abc
import gym
from src.cartpole.agent import Agent


class AgentFactory(metaclass=abc.ABCMeta):
    @staticmethod
    def new(env: gym.Env) -> Agent:
        raise NotImplementedError()
