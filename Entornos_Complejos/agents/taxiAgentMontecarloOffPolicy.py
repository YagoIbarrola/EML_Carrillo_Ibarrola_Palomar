import numpy as np
import gymnasium as gym
from agents.agent import Agent


class TaxiAgentMontecarloOffPolicy(Agent):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.4,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.01,
        final_epsilon: float = 0.01,
        discount_factor: float = 1.0,
    ):
        # Llamamos al constructor de la clase padre
        super().__init__(
            env=env,
            epsilon=epsilon,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor
        )

        # Inicializamos la tabla Q y la matriz C (pesos acumulados)
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.C = np.zeros([env.observation_space.n, env.action_space.n])

        # Preparamos una memoria temporal para almacenar las transiciones
        self.episode = []

    def get_action(self, obs) -> int:
        """
        Política de comportamiento (b): epsilon-greedy.
        """
        # Exploramos tomando una acción aleatoria con probabilidad epsilon
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # Explotamos eligiendo la mejor acción conocida
        else:
            return int(np.argmax(self.Q[obs, :]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs
    ):
        """
        Adaptación de Monte Carlo Off-Policy para la interfaz paso a paso.
        """
        # Guardamos la transición actual en nuestra memoria
        self.episode.append((obs, action, reward))

        # Ejecutamos el aprendizaje Off-Policy solo cuando el episodio termina
        if terminated:
            # Hacemos una copia de la tabla Q para calcular el error al finalizar
            old_q = np.copy(self.Q)

            G = 0.0
            W = 1.0
            
            # Recorremos el episodio hacia atrás
            for state, act, rew in reversed(self.episode):
                G = rew + self.discount_factor * G
                
                # Actualizamos la matriz de pesos acumulados C
                self.C[state, act] += W
                
                # Actualizamos el valor Q usando la ponderación W/C
                self.Q[state, act] += (W / self.C[state, act]) * (G - self.Q[state, act])
                
                # Obtenemos la acción óptima según nuestra política objetivo (pi)
                best_action = int(np.argmax(self.Q[state, :]))
                
                # Rompemos el bucle si la acción tomada no coincide con la óptima
                if act != best_action:
                    break
                    
                # Calculamos la probabilidad de nuestra política de comportamiento (b)
                prob_b = (1.0 - self.epsilon) + (self.epsilon / self.env.action_space.n)
                
                # Ajustamos el peso W multiplicando por pi(A|S) / b(A|S)
                W = W * (1.0 / prob_b)

            # Calculamos el error de entrenamiento y lo guardamos
            error = np.max(np.abs(self.Q - old_q))
            self.training_error.append(error)

            # Limpiamos la memoria para el siguiente episodio
            self.episode = []