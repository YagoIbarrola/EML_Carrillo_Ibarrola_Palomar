from collections import defaultdict
import gymnasium as gym
import numpy as np
from agents.taxiAgentSARSA import TaxiAgentSARSA

class TaxiAgentExpectedSARSA(TaxiAgentSARSA):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Expected SARSA agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """

        super().__init__(
            env,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
        )

    def get_action_probabilities(self, obs: int) -> np.ndarray:
        """Calculate the probability of taking each action in the given state under the current policy."""
        #Every action has a base probability of epsilon / num_actions (exploration)
        action_probabilities = np.ones(self.env.action_space.n) * self.epsilon / self.env.action_space.n
        #Best action has an additional probability of (1 - epsilon) (exploitation)
        best_action = int(np.argmax(self.q_values[obs]))
        action_probabilities[best_action] += (1.0 - self.epsilon)
        return action_probabilities

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Update Q-value based on experience.

        This is the heart of Expected SARSA: learn from (state, action, reward, next_state) tuples.
        """
        # Compute expected value of next state's actions (if episode not terminated)
        future_q_value = (not terminated) * np.dot(self.get_action_probabilities(next_obs), self.q_values[next_obs])
        # What should the current Q-value be based on next step?
        target = reward + self.discount_factor * future_q_value
        # How far off is our current estimate?
        temporal_difference = target - self.q_values[obs][action]
        # Update our estimate in the direction of the error
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)