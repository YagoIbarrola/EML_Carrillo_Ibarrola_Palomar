from collections import defaultdict
import gymnasium as gym
import numpy as np
from agents.agent import Agent

class TaxiAgentSARSA(Agent):

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a SARSA agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """

        super().__init__(
            env=env,
            epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        #Additional attributes
        self.lr = learning_rate
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs: int) -> tuple[int, bool]:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: An integer representing the chosen action.
            is_exploring: A boolean indicating if the action was exploratory.
        """
        # Exploration
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), True

        # Exploitation
        return int(np.argmax(self.q_values[obs])), False

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
        next_action: int,
    ):
        """Update Q-value based on experience.

        Learn from (state, action, reward, next_state, next_action) tuples.
        """
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]
        # What should the current Q-value be based on next step?
        target = reward + self.discount_factor * future_q_value
        # How far off is our current estimate?
        temporal_difference = target - self.q_values[obs][action]
        # Update our estimate in the direction of the error
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def get_current_policy(self):
        """
        Extrae la política actual del agente evaluando el mejor Q-value para todos los estados posibles.
        """
        # Creamos un array vacío de tamaño 500 para guardar la mejor acción de cada estado
        n_states = 500
        policy = np.zeros(n_states, dtype=int)

        for state in range(n_states):
            # np.argmax nos da el índice de la acción con el valor más alto.
            # Al consultar agent.q_values[state], el defaultdict creará una entrada de ceros 
            # automáticamente si el estado nunca fue visitado (devolviendo 0 por defecto).
            policy[state] = int(np.argmax(self.q_values[state]))
        return policy