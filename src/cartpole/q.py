from typing import List, Dict, Tuple, Callable, Union
import os
import collections
import random
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from src.cartpole.agent import Agent
from src.cartpole.agent_factory import AgentFactory
from src.cartpole.lr_exponential_decay import LRExponentialDecay


class FeatureTransformer:

    def __init__(self):
        self._num_features = 2  # angle & velocity
        self._num_states = self.num_bins ** self._num_features
        self._state_space = np.zeros((self._num_states))
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                self._state_space[(i * self.num_bins) + j] = int("{}{}".format(i, j))
        return

    @staticmethod
    def build_state(features) -> int:
        return int("".join(map(lambda feature: str(int(feature)), features)))

    @property
    def num_bins(self) -> int:
        return 10

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def state_space(self) -> np.ndarray:
        """
        All possible discrete states
        :return:
        """
        return self._state_space


class SimpleAgent:
    """
    Agent using Q Learning with continuous feature space discretized into 10 bins per feature
    """

    def __init__(self):
        # Hyper Param
        self._epsilon = 1.0
        self._gamma = 0.9

        # Environment
        self._feature_transformer = FeatureTransformer()
        self._actions = list(range(0, 2))
        self._num_actions = len(self._actions)
        self._num_states = self._feature_transformer.num_states
        self._bin_cnt = np.zeros((self._num_states, self._num_actions))

        self._global_iter = 1  # avoid div by zero
        self._master_iter = 1

        # Q Calc
        self._q_learning_rate = 10e-3
        self._q = np.random.uniform(low=-1, high=1, size=(self._num_states, self._num_actions))

        # Q Model Calc
        self._main_model = self._create_model()
        self._replay_buffer = collections.deque(maxlen=5000)
        self._lr_decay = LRExponentialDecay(num_epoch=3000, initial_lr=0.0075, min_lr=0.0005, decay_rate=0.01)

        # Visualise
        mpeg_writer = animation.writers['ffmpeg']
        self._writer = mpeg_writer(fps=15, metadata=dict(artist='e2pii-1'), bitrate=1800)
        self._fig_num = 1
        # self._animate()
        return

    def init(self,
             env) -> None:
        """
        Called once at the start of a session
        """
        return

    def _predict_q(self,
                   discrete_state: int) -> np.float:
        """
        Predict (get) the Q Values for each action and select the greedy action
        :param discrete_state: The current environment state
        :return: The current Q values from the Q Grid
        """
        return self._q[discrete_state]

    def _predict_m(self,
                   discrete_state: int) -> np.float:
        """
        Predict the Q Values for each action and select the greedy action
        :param discrete_state: The current environment state
        :return: The Q Values for the given state as predicted by the NN Model.
        """
        return self._main_model.predict(x=np.array([discrete_state]).reshape((1, 1)))

    def _q_calc(self,
                reward: float,
                state_predicted: float,
                q_learning_rate: float = None) -> np.float:
        """
        Calculate the updated Q Value given the SARS' event
        """
        if q_learning_rate is None:
            q_learning_rate = self._q_learning_rate
        G = (reward - state_predicted)
        return q_learning_rate * G

    def _update_q_grid(self,
                       samples: List) -> None:
        """
        Update the Q grid for a random sample.
        """
        for discrete_state, action, reward in samples:
            state_pred = self._q[discrete_state]
            self._q[discrete_state, action] += self._q_calc(reward, state_pred[action])
            # self._q[discrete_state, action] = reward
        return

    def _update_model_q(self,
                        samples: List,
                        sample_size) -> None:
        """
        Train the NN model given a sample set of events from the replay buffer.
        """
        discrete_states = np.empty((sample_size, 1), dtype=np.float)
        actions = np.empty((sample_size), dtype=np.int)
        rewards = np.empty((sample_size), dtype=np.float)

        i = 0
        # We need to predict in batches to reduce model call overhead.
        for discrete_state, action, reward in samples:
            discrete_states[i] = discrete_state
            actions[i] = action
            rewards[i] = reward
            i += 1

        discrete_state_preds = self._main_model.predict(discrete_states)

        x_train = discrete_states
        y_train = discrete_state_preds

        # Calc Q Updates
        for i in range(sample_size):
            y_train[i, actions[i]] += self._q_calc(rewards[i],
                                                   discrete_state_preds[i][actions[i]],
                                                   q_learning_rate=.1)
            i += 1

        self._lr_decay.update(self._master_iter)
        print("start fit {} ".format(self._master_iter))
        history = self._main_model.fit(x_train, y_train, epochs=20, verbose=0,
                                       callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                           self._lr_decay.lr)])
        print("end fit")

        return

    def _create_model(self) -> tf.keras.Model:
        """
        Keras (TF) model that will learn and predict the value of a given state.
        Input Shape = (1,) = The discrete state output from feature transformer
        Output Shape = (2) = Predicted Q Value for both Left and Right Actions.
        :return: A Keras Model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(input_shape=(1,), units=1, name='Input'),
            tf.keras.layers.Dense(25, activation=tf.nn.relu, name='dense1'),
            tf.keras.layers.Dense(100, activation=tf.nn.relu, name='dense2'),
            tf.keras.layers.Dense(400, activation=tf.nn.relu, name='dense3'),
            tf.keras.layers.Dense(100, activation=tf.nn.relu, name='dense4'),
            tf.keras.layers.Dense(25, activation=tf.nn.relu, name='dense5'),
            tf.keras.layers.Dense(self._num_actions, name='output')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0075),
            loss=tf.keras.losses.mean_squared_error
        )
        return model

    def reward(self,
               state: np.ndarray,
               action: int,
               reward: float,
               train_every: int = 1,
               sample_size: int = 250) -> None:
        """
        Update Q with respect to given reward for State/Action pair
        :param state: The state the action was taken in
        :param action: The action taken
        :param reward: The reward for the State/Action
        """
        discrete_state = FeatureTransformer.build_state(state)

        # Update replay buffer
        self._replay_buffer.append([discrete_state, action, reward])

        if self._master_iter % train_every != 0 or len(self._replay_buffer) < sample_size:
            self._master_iter += 1
            return

        samples = random.sample(self._replay_buffer, sample_size)

        # Q Value calculation
        self._update_q_grid(samples)

        # Q Model update
        self._update_model_q(samples, sample_size)

        # Explanation & visualization
        # self._bin_cnt[discrete_prev_state] += 1
        if self._master_iter == 1 or self._master_iter % 10 == 0:  # self._evry == 0:
            print("LR: {}".format(self._lr_decay.lr(None)))
            ax, ql, qr = self._q2mesh()
            _, qml, qmr = self._qm2mesh()
            self._visualise2(self._master_iter, ax, ql, qr, qml, qmr)
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
        self._animate()
        return

    def debug(self) -> Dict:
        return {"epsilon": self._epsilon}

    def _q2mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the q values as a mesh for plotting.
        :return: q values as mesh Tuple(bins, left action q vals, right action q values)
        """
        qmeshl = np.zeros((self._feature_transformer.num_bins, self._feature_transformer.num_bins))
        qmeshr = np.zeros((self._feature_transformer.num_bins, self._feature_transformer.num_bins))
        bins = range(self._feature_transformer.num_bins)
        for i in bins:  # pole_vel
            for j in bins:  # pole_angle
                qmeshl[i, j] = self._q[int("{}{}".format(i, j))][0]
                qmeshr[i, j] = self._q[int("{}{}".format(i, j))][1]
        return np.array(bins), qmeshl, qmeshr

    def _qm2mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the q values as a mesh for plotting where q is predicted by the model
        :return: q values as mesh Tuple(bins, left action q vals, right action q values)
        """
        mq = self._main_model.predict(self._feature_transformer.state_space)
        qmeshl = np.zeros((self._feature_transformer.num_bins, self._feature_transformer.num_bins))
        qmeshr = np.zeros((self._feature_transformer.num_bins, self._feature_transformer.num_bins))
        bins = range(self._feature_transformer.num_bins)
        for i in bins:  # pole_vel
            for j in bins:  # pole_angle
                qmeshl[i, j] = mq[int("{}{}".format(i, j))][0]
                qmeshr[i, j] = mq[int("{}{}".format(i, j))][1]
        return np.array(bins), qmeshl, qmeshr

    def _bin2mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the bin counts as a mesh for plotting
        :return: bin count values as mesh Tuple(bins, count values)
        """
        bin_mesh = np.zeros((self._feature_transformer.num_bins, self._feature_transformer.num_bins))
        bins = range(self._feature_transformer.num_bins)
        for i in bins:  # pole_vel
            for j in bins:  # pole_angle
                bin_mesh[i, j] = self._bin_cnt[int("{}{}".format(i, j))][0]
        return np.array(bins), bin_mesh

    def _visualise2(self,
                    iteration: int,
                    ax: np.ndarray,
                    ql: np.ndarray,
                    qr: np.ndarray,
                    ql_m: np.ndarray,
                    qr_m: np.ndarray) -> None:
        """
        Render the visual plot of the current state
        :param iteration: The iteration that this visualisation corresponds to
        :param ax: The axis bins (all features have same bins so only need once)
        :param ql: The Q Values by bin for the left action
        :param qr: The Q Values by bin for the right action
        :param ql: The Q Values by bin for the left action as predicted by the model
        :param qr: The Q Values by bin for the right action as predicted by the model
        """
        fig = plt.figure()
        fig.suptitle("Iteration {}".format(iteration))
        ax1 = plt.subplot(221)
        s1 = ax1.contourf(ax, ax, ql, 20, cmap=cm.coolwarm, antialiased=False)
        ax2 = plt.subplot(222)
        s2 = ax2.contourf(ax, ax, qr, 20, cmap=cm.coolwarm, antialiased=False)
        ax3 = plt.subplot(223)
        s3 = ax3.contourf(ax, ax, ql_m, 20, cmap=cm.viridis, antialiased=False)
        ax4 = plt.subplot(224)
        s4 = ax4.contourf(ax, ax, qr_m, 20, cmap=cm.viridis, antialiased=False)
        fig.colorbar(s1, ax=ax1)
        fig.colorbar(s2, ax=ax2)
        fig.colorbar(s3, ax=ax3)
        fig.colorbar(s4, ax=ax4)
        ax1.set_title("Q Left")
        ax1.set_ylabel("Pole Angle")
        ax2.set_title("Q Right")
        ax3.set_title("Q Left Model")
        ax3.set_ylabel("Pole Angle")
        ax3.set_xlabel("Pole Velocity")
        ax4.set_title("Q Right Model")
        ax4.set_xlabel("Pole Velocity")
        fig.tight_layout()
        plt.savefig("./images/cartpole_{}.png".format(self._fig_num), pad_inches=0.2)
        self._fig_num += 1
        plt.show()
        return

    def _animate(self) -> None:
        """
        Convert the saved plots into a animation (movie)
        :return:
        """
        anim_fig = plt.figure()
        plt.axis('off')
        fig_num = 1
        plots = []
        while os.path.exists("./images/cartpole_{}.png".format(fig_num)):
            img = mgimg.imread("./images/cartpole_{}.png".format(fig_num))
            imgplot = plt.imshow(img)
            plots.append([imgplot])
            fig_num += 1

        anim = animation.ArtistAnimation(anim_fig, plots, interval=3000, blit=True, repeat_delay=1000)
        anim.save("./images/mov.mp4", writer=self._writer)
        plt.axis('on')
        return

    @property
    def _evry(self) -> int:
        """
        Intervals at which we generate updated visualizations. More frequent in early stages
        as there is more noticeable change before the q values start to converge.
        :return: The update 'every' interval
        """
        return 1000


if __name__ == "__main__":
    rewards = [[1, 5, 10, 10, 25, 50, 100, 50, 25, 10],
               [5, 10, 25, 25, 50, 100, 200, 100, 50, 25],
               [10, 25, 25, 50, 100, 200, 300, 200, 100, 50],
               [10, 25, 50, 50, 50, 100, 200, 100, 50, 25],
               [25, 50, 100, 50, 25, 50, 100, 50, 25, 10],
               [50, 100, 200, 100, 50, 50, 50, 25, 10, 5],
               [100, 200, 300, 200, 100, 50, 25, 25, 10, 5],
               [50, 100, 200, 100, 50, 25, 25, 10, 5, 1],
               [25, 50, 100, 50, 25, 10, 10, 5, 1, 1],
               [10, 25, 50, 25, 10, 5, 5, 1, 1, 1]]
    sm = SimpleAgent()
    print("Start")
    for i in range(0, 10000):
        st = np.random.randint(0, 10, 2)
        actn = np.random.randint(0, 2, 1)[0]
        rw = rewards[st[0]][st[1]]
        if actn > 0:
            rw = -rw
        sm.reward(state=st, action=actn, reward=rw)
        if i % 500 == 0:
            print(i)
    print("Done")
    sm.final()
    exit(0)
