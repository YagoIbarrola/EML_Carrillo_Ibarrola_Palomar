from collections import defaultdict
import gymnasium as gym
import numpy as np
from agents.agent import Agent

class LunarAgentSARSA(Agent):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99, # Generalmente 0.99 para LunarLander
        decay_type: str = "linear",
    ):
        super().__init__(
            env=env,
            epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            decay_type=decay_type,
        )

        # Ajustamos el learning rate por la cantidad de tilings (práctica recomendada)
        # Buscamos n_tilings en la cadena de wrappers
        n_tilings = getattr(env, "n_tilings", None)
        if n_tilings is None:
            # Si no está en la capa exterior, buscamos en el entorno envuelto
            n_tilings = getattr(env.unwrapped, "n_tilings", 8) # 8 por defecto

        self.lr = learning_rate / n_tilings 
        self.n_actions = env.action_space.n
        
        # Matriz de pesos W: [total_features, n_actions]
        self.w = np.zeros((env.total_features, self.n_actions))
        self.training_error = []

    def q_value(self, active_features, action):
        """Calcula Q(s,a) sumando los pesos de las características activas."""
        return self.w[active_features, action].sum()

    def get_action(self, active_features: list[int]) -> tuple[int, bool]:
        """Epsilon-greedy con función de aproximación lineal."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), True

        # Exploitation: evaluamos Q(s,a) para todas las acciones
        q_values = [self.q_value(active_features, a) for a in range(self.n_actions)]
        
        # Empates (argmax aleatorio)
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions), False

    def update(
        self,
        obs: list[int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: list[int],
        next_action: int,
    ):
        """Regla de actualización Semigradiente SARSA."""
        # Calculamos Q actual
        q_current = self.q_value(obs, action)
        
        # Calculamos el Q futuro
        if terminated:
            q_next = 0.0
        else:
            q_next = self.q_value(next_obs, next_action)
            
        target = reward + self.discount_factor * q_next
        temporal_difference = target - q_current
        
        # Actualizamos SOLO los pesos de los tiles activos
        for feature_idx in obs:
            self.w[feature_idx, action] += self.lr * temporal_difference

        self.training_error.append(temporal_difference)

    def get_current_policy(self):
        """
        Return the current policy (e.g., action with highest Q-value for each state).
        Must be implemented by subclasses.
        """
        return # np.argmax(self.w, axis=1)  # Política greedy basada en los pesos actuales