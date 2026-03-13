import numpy as np
import gymnasium as gym
from agents.agent import Agent

class TaxiAgentMontecarloOffPolicy(Agent):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.4,
        epsilon_decay: float = 0.01,
        final_epsilon: float = 0.01,
        discount_factor: float = 1.0,
        decay_type: str = "linear",
    ):
        super().__init__(
            env=env,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            decay_type=decay_type
        )

        # Inicializamos las estructuras de datos (Q, C y pi)
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.C = np.zeros([env.observation_space.n, env.action_space.n])
        
        # Guardamos la politica optima de forma explicita (pi)
        self.pi = np.zeros(env.observation_space.n, dtype=int)
        
        # Asignamos una accion aleatoria inicial para cada estado rompiendo el empate a cero
        for s in range(env.observation_space.n):
            self.pi[s] = self._random_argmax(self.Q[s])

        # Preparamos nuestra memoria temporal para almacenar el episodio
        self.episode = []
        
    def _random_argmax(self, q_values: np.ndarray) -> int:
        # Encontramos el valor maximo actual
        max_val = np.max(q_values)
        
        # Extraemos todos los indices que comparten exactamente ese valor maximo
        indices_empate = np.flatnonzero(np.isclose(q_values, max_val))
        
        # Elegimos uno de los indices empatados al azar
        return int(np.random.choice(indices_empate))
    
    def get_action(self, obs) -> tuple[int, bool]:
        """
        Politica de comportamiento (b): Cualquier politica suave (soft policy).
        Usamos epsilon-greedy basandonos en nuestra politica objetivo pi.
        """
        # Exploramos tomando una accion completamente aleatoria
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), True
        # Explotamos tomando la accion que dicta nuestra politica optima
        else:
            return int(self.pi[obs]), False


    def get_current_policy(self):
        """
        Devolvemos la politica actual (pi) que es la accion con el mayor valor Q para cada estado.
        """
        return self.pi

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs
    ):
        """
        Implementacion estricta de Off-Policy MC control segun la literatura.
        """
        # Anadimos el paso actual a nuestra memoria del episodio
        self.episode.append((obs, action, reward))

        # Procesamos todo el episodio una vez finalizado
        if terminated:
            # Hacemos una copia de nuestra tabla Q para calcular el error posteriormente
            old_q = np.copy(self.Q)

            G = 0.0
            W = 1.0
            
            # Recorremos el episodio desde el estado final hasta el estado inicial
            for state, act, rew in reversed(self.episode):
                G = rew + self.discount_factor * G
                
                # Sumamos el peso al acumulador
                self.C[state, act] += W
                
                # Actualizamos nuestra estimacion de la tabla Q
                self.Q[state, act] += (W / self.C[state, act]) * (G - self.Q[state, act])
                
                # Actualizamos nuestra politica objetivo para este estado concreto
                self.pi[state] = self._random_argmax(self.Q[state, :])
                
                # Comprobamos si la accion tomada en el episodio diverge de nuestra politica optima
                if act != self.pi[state]:
                    break
                    
                # Calculamos la probabilidad de tomar esta accion bajo nuestra politica de comportamiento (b)
                prob_b = (1.0 - self.epsilon) + (self.epsilon / self.env.action_space.n)
                
                # Ajustamos el peso multiplicandolo por 1 / b(A_t | S_t)
                W = W * (1.0 / prob_b)

            # Calculamos el error maximo de entrenamiento para este paso
            error = np.max(np.abs(self.Q - old_q))
            self.training_error.append(error)

            # Vaciamos nuestra memoria de transiciones para el siguiente ciclo
            self.episode = []