from collections import defaultdict
from agents.agent import Agent
import gymnasium as gym
import numpy as np


class TaxiAgentQLearning(Agent):

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        decay_type: str = "linear",
    ):
        super().__init__(
            env=env,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            decay_type=decay_type,
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


    def get_current_policy(self):
        """
        Extrae la política actual del agente evaluando el mejor Q-value para todos los estados posibles.
        """
        # Creamos un array vacío de tamaño 500 para guardar la mejor acción de cada estado
        n_states = self.env.observation_space.n
        policy = np.zeros(n_states, dtype=int)

        for state in range(n_states):
            # np.argmax nos da el índice de la acción con el valor más alto.
            # Al consultar agent.q_values[state], el defaultdict creará una entrada de ceros 
            # automáticamente si el estado nunca fue visitado (devolviendo 0 por defecto).
            policy[state] = int(np.argmax(self.q_values[state]))
        return policy