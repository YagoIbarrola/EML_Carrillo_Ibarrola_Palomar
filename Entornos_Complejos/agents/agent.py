from abc import ABC, abstractmethod
from collections import defaultdict
import gymnasium as gym
import numpy as np


class Agent(ABC):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        """
        Base class for tabular RL agents.

        Args:
            env: The training environment
            learning_rate: Step size for updates (0-1)
            initial_epsilon: Initial exploration rate
            epsilon_decay: Epsilon decrease per episode
            final_epsilon: Minimum exploration rate
            discount_factor: Gamma (0-1)
        """
        self.env = env
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = epsilon
        self.inital_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs):
        """
        Epsilon-greedy action selection.
        """
        pass

    @abstractmethod
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """
        Learning rule (algorithm-specific).
        Must be implemented by subclasses.
        """
        pass

    def decay_epsilon(self):
        """
        Reduce exploration rate after each episode.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)