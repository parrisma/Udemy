import sys
from typing import Tuple
import gym
import numpy as np
import matplotlib.pyplot as plt
from src.cartpole.agent_factory import AgentFactory, Agent
from src.cartpole.random_agent import RandomAgentFactory
from src.cartpole.linear_model_agent import LinearModelAgentFactory
from src.cartpole.q_model_agent import QModelAgentFactory

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


class CartPole1:
    _env: gym.Env
    _agent: Agent
    _history: np.ndarray
    _rewards: np.ndarray

    def __init__(self,
                 agent_factory: AgentFactory):
        self._env = gym.make('CartPole-v0')
        self._env.reset()
        self._agent = agent_factory.new(self._env)
        self._rewards = None  # noqa
        self._history = None  # noqa
        return

    def _play_episode(self) -> Tuple[int, float]:
        """
        Play a single episode
        :return: number of steps & total reward
        """
        done = False
        num_steps = 0
        total_reward = 0
        state = self._env.reset()
        self._agent.reset()
        while not done and num_steps < 200:  # 200 is max episodes allowed by CartPole
            action = self._agent.chose_action(state)
            prev_state = state
            state, reward, done, _ = self._env.step(action)
            total_reward += reward
            if done and num_steps < 199:
                self._agent.reward(state, prev_state, action, -300)  # Q Value penalty for nto reaching max steps
            else:
                self._agent.reward(state, prev_state, action, reward)
            num_steps += 1
        return num_steps, total_reward

    def run(self,
            num_episodes: int) -> None:
        """
        Execute the given number of episodes
        :param num_episodes: The number of episodes to play
        :return: The episode length history
        """
        self._history = np.zeros(num_episodes)
        self._rewards = np.zeros(num_episodes)
        for e in range(num_episodes):
            n, total_reward = self._play_episode()
            self._history[e] = n
            self._rewards[e] = total_reward
            self._agent.done()
            if e % 100 == 0 and e > 0:
                self._summary_100(e)
        self._agent.final()
        self._summary_final()
        return

    def _summary_100(self,
                     episode_number: int) -> None:
        mean_reward = self._rewards[episode_number - 100:episode_number].mean()
        mn, mx, mean_steps, stdd = Util.stats(self._history[:episode_number])
        print("episode : [{}] reward [{:7.3f}] reward [{:7.3f}] Steps Ave:[{:12.6f}] eps :[{:12.6f}]".format(
            episode_number,
            self._rewards[episode_number],
            mean_reward,
            mean_steps,
            self._agent.debug()['epsilon']))
        return

    def _plot(self):
        n = len(self._rewards)
        running_avg = np.empty(n)
        for t in range(n):
            running_avg[t] = self._rewards[max(0, t - 100):(t + 1)].mean()
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()

    def _summary_final(self) -> None:
        """
        Print / Plot summary of overall run.
        :return:
        """
        print("Min [{:12.6f}] Max [{:12.6f}] Ave:[{:12.6f}] StdDev:[{:12.6f}]".format(*Util.stats(self._history)))
        self._plot()
        return


if __name__ == "__main__":
    np.random.seed(42)
    if 'random' in sys.argv:
        CartPole1(RandomAgentFactory()).run(100)
    elif 'linear' in sys.argv:
        CartPole1(LinearModelAgentFactory()).run(5000)
    else:
        CartPole1(QModelAgentFactory()).run(10000)
