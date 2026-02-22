
from collections import defaultdict
from agents.agent import Agent
import gymnasium as gym
import numpy as np

class TaxiAgentMontecarloOnPolicy(Agent):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.4,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.01,
        final_epsilon: float = 0.01,
        discount_factor: float = 1.0,
    ):
        # Inicializamos la clase padre llamando a super()
        super().__init__(
            env=env,
            epsilon=epsilon,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        # Inicializamos la tabla Q y el contador de visitas (N)
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.n_visits = np.zeros([env.observation_space.n, env.action_space.n])
        self.episode = []  # Para almacenar las transiciones del episodio actual

    def get_action(self, obs) -> int:
        """
        Política epsilon-greedy: 
        Con probabilidad epsilon, exploramos tomando una acción aleatoria.
        Con probabilidad (1 - epsilon), explotamos tomando la mejor acción conocida.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.Q[obs, :]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """
        Adaptación de Monte Carlo para la interfaz paso a paso.
        Acumulamos las transiciones y actualizamos los valores Q solo al finalizar.
        """
        self.episode.append((obs, action, reward))
        # Si el episodio termina, ejecutamos el algoritmo de Monte Carlo
        if terminated:
            # Guardamos una copia exacta de la tabla Q antes de aprender
            old_q = np.copy(self.Q)

            G = 0.0
            # Recorremos el episodio hacia atrás para calcular el retorno G
            for state, act, rew in reversed(self.episode):
                G = rew + self.discount_factor * G
                self.n_visits[state, act] += 1.0
                
                # Calculamos la tasa de aprendizaje incremental
                alpha = 1.0 / self.n_visits[state, act]
                
                # Actualizamos el valor Q
                self.Q[state, act] += alpha * (G - self.Q[state, act])

            # Calculamos el error de entrenamiento (diferencia máxima)
            error = np.max(np.abs(self.Q - old_q))
            self.training_error.append(error)
            
            self.episode = []