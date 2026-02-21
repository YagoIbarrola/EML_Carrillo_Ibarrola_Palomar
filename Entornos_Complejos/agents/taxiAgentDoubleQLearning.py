from collections import defaultdict
from agents.agent import Agent
import gymnasium as gym
import numpy as np


class TaxiAgentDoubleQLearning(Agent):

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        super().__init__(
            env,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
        )

        # Two Q-tables
        self.lr = learning_rate
        self.q1_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.q2_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs):
        # Used by epsilon-greedy
                # Exploration
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Exploitation
        q_values = self.q1_values[obs] + self.q2_values[obs]
        return int(np.argmax(q_values))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):

        if np.random.random() < 0.5:
            # Update Q1
            if terminated:
                target = reward
            else:
                best_action = np.argmax(self.q1_values[next_obs])
                target = reward + self.discount_factor * \
                         self.q2_values[next_obs][best_action]

            td_error = target - self.q1_values[obs][action]
            self.q1_values[obs][action] += self.lr * td_error

        else:
            # Update Q2
            if terminated:
                target = reward
            else:
                best_action = np.argmax(self.q2_values[next_obs])
                target = reward + self.discount_factor * \
                         self.q1_values[next_obs][best_action]

            td_error = target - self.q2_values[obs][action]
            self.q2_values[obs][action] += self.lr * td_error

        self.training_error.append(td_error)