from collections import defaultdict
from agents.agent import Agent
import gymnasium as gym
import numpy as np


class TaxiAgentDoubleQLearning(Agent):

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

        # Two Q-tables
        self.lr = learning_rate
        self.q1_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.q2_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs):
        '''
        Choose A with from S using epsilon-greedy policy in Q1 + Q2:
        '''
        # Used by epsilon-greedy
                # Exploration
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), True

        # Exploitation
        q_values = self.q1_values[obs] + self.q2_values[obs]
        return int(np.argmax(q_values)), False

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        ''' 
        With 0.5 Q_1(S,A) <- Q_1(S,A) + alpha(R + gamma Q_2(S', argmax_a Q_1(S',a)) - Q_1(S,A))
        With 0.5 Q_2(S,A) <- Q_2(S,A) + alpha(R + gamma Q_1(S', argmax_a Q_2(S',a)) - Q_2(S,A))
        '''
        if np.random.random() < 0.5:
            # Update Q1
            if terminated:
                target = reward
            else:
                best_action = np.argmax(self.q1_values[next_obs])
                target = reward + self.discount_factor * self.q2_values[next_obs][best_action]

            td_error = target - self.q1_values[obs][action]
            self.q1_values[obs][action] += self.lr * td_error

        else:
            # Update Q2
            if terminated:
                target = reward
            else:
                best_action = np.argmax(self.q2_values[next_obs])
                target = reward + self.discount_factor * self.q1_values[next_obs][best_action]

            td_error = target - self.q2_values[obs][action]
            self.q2_values[obs][action] += self.lr * td_error

        self.training_error.append(td_error)
    
    def get_current_policy(self):
        """
        Extract the current policy by evaluating the best sum of Q1 and Q2 values for all possible states.
        """
        # Creamos un array vacío de tamaño 500 para guardar la mejor acción de cada estado
        n_states = self.env.observation_space.n
        policy = np.zeros(n_states, dtype=int)

        for state in range(n_states):
            # np.argmax nos da el índice de la acción con el valor más alto.
            # Al consultar agent.q_values[state], el defaultdict creará una entrada de ceros 
            # automáticamente si el estado nunca fue visitado (devolviendo 0 por defecto).
            policy[state] = int(np.argmax(self.q1_values[state] + self.q2_values[state]))
        return policy