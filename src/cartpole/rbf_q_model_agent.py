from typing import Dict
import numpy as np
import gym
from src.cartpole.agent import Agent
from src.cartpole.agent_factory import AgentFactory
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class SGDRegressor:
    def __init__(self,
                 dimension: int,
                 learning_rate: float):
        self.weights = np.random.randn(dimension) / np.sqrt(dimension)
        self.learning_rate = learning_rate

    def partial_fit(self, x, y) -> None:
        self.weights += self.learning_rate * (y - x.dot(self.weights)).dot(x)
        return

    def predict(self, x):
        return x.dot(self.weights)


class FeatureTransformer:
    def __init__(self):
        # https://www.youtube.com/watch?v=Qc5IyLW_hns
        # https://stats.stackexchange.com/questions/317391/gamma-as-inverse-of-the-variance-of-rbf-kernel
        # https://www.youtube.com/watch?v=m2a2K4lprQw
        # observation_examples = np.array([env.observation_space.sample() for x in range(20000)])
        # NOTE!! state samples are poor, b/c you get velocities --> infinity
        observation_examples = np.random.random((20000, 4)) * 2 - 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to convert a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        # (this would also seem to have the effect of smoothing the decision boundary by balancing the
        #  influence of points at different 'distances' from it)
        #
        # This is 'like' a single layer in a NN where input is (4,) and hidden layer is (4000) and where
        # RBF is the non-linearity
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),  # 'distant' parts of the feature space
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000))  # 'near' parts of the feature space.
        ])
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class RBFQModelAgent(Agent):
    """
    Agent using Q Learning with continuous feature space transformed using RBF's
    """
    _feature_transformer: FeatureTransformer
    _learning_rate: float
    _epsilon: float
    _gamma: float

    def __init__(self,
                 env: gym.Env):
        self._feature_transformer = FeatureTransformer()
        self._learning_rate = 0.1
        self._epsilon = 1.0
        self._gamma = 0.99
        self._global_iter = 1  # avoid div by zero
        self._actions = list(range(0, env.action_space.n))
        self._num_actions = len(self._actions)
        self._models = []
        #
        # a model for each action 'like' multiple outputs on a NN.
        #
        for i in range(env.action_space.n):
            model = SGDRegressor(self._feature_transformer.dimensions, self._learning_rate)
            self._models.append(model)
        return

    def init(self,
             env) -> None:
        """
        Called once at the start of a session
        """
        return

    def _predict(self,
                 state: np.ndarray) -> int:
        """
        Predict the Q Values for each action and select the greedy action
        :param state: The current environment state
        :return: The greedy action
        """
        X = self._feature_transformer.transform(np.atleast_2d(state))
        result = np.stack([m.predict(X) for m in self._models]).T
        return result

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
        trans_state = self._feature_transformer.transform(np.atleast_2d(prev_state))
        self._models[action].partial_fit(trans_state, [G])  # single step update
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


class RBFQModelAgentFactory(AgentFactory):
    @staticmethod
    def new(env: gym.Env) -> Agent:
        return RBFQModelAgent(env)
