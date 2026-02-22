from collections import defaultdict
import gymnasium as gym
import numpy as np
from agents.agent import Agent

class TaxiAgentQLearning(Agent):

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        super().__init__(
            env,
            learning_rate,
            epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
        )

        # Single Q-table
        self.lr = learning_rate
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))


    def get_action(self, obs):
        """
        Required by BaseAgent for epsilon-greedy.
        """
        # Exploration
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), True

        # Exploitation
        q_values = self.q_values[obs]
        return int(np.argmax(q_values)), False

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """
        Standard Q-Learning update:
        Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
        """

        if terminated:
            future_q_value = 0.0
        else:
            future_q_value = np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value
        td_error = target - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * td_error
        self.training_error.append(td_error)