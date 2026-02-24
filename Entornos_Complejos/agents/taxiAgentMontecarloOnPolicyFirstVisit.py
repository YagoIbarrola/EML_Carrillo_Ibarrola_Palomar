import numpy as np
import gymnasium as gym
from agents.agent import Agent

class TaxiAgentMontecarloOnPolicyFirstVisit(Agent):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.1,  # El algoritmo de la imagen asume un epsilon pequeño y fijo
        epsilon_decay: float = 0.0,
        final_epsilon: float = 0.1,
        discount_factor: float = 1.0,
    ):
        # Inicializamos la clase padre
        super().__init__(
            env=env,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        num_estados = env.observation_space.n
        num_acciones = env.action_space.n

        # Q(s, a) inicializado a cero (o arbitrariamente)
        self.Q = np.zeros([num_estados, num_acciones])
        
        # Para calcular average(Returns(s,a)) sin agotar la memoria
        self.n_visits = np.zeros([num_estados, num_acciones])
        
        # pi <- an arbitrary epsilon-soft policy
        # Inicializamos con probabilidad uniforme para todas las acciones
        self.pi = np.ones([num_estados, num_acciones]) / num_acciones
        
        # Memoria del episodio
        self.episode = []

    def _random_argmax(self, q_values: np.ndarray) -> int:
        # Rompemos empates arbitrariamente (with ties broken arbitrarily)
        max_val = np.max(q_values)
        indices_empate = np.flatnonzero(np.isclose(q_values, max_val))
        return int(np.random.choice(indices_empate))

    def get_action(self, obs) -> tuple[int, bool]:
        # Generamos un episodio siguiendo la politica pi (Generate an episode following pi)
        # Elegimos una accion basandonos en las probabilidades almacenadas en pi(a|S_t)
        accion = np.random.choice(self.env.action_space.n, p=self.pi[obs])
        
        # Determinamos si la accion fue de exploracion para las graficas
        es_exploracion = (accion != np.argmax(self.Q[obs]))
        return int(accion), es_exploracion

    def get_current_policy(self):
        return np.argmax(self.Q, axis=1)

    def update(
        self,
        obs,
        action: int,
        reward: float,
        done: bool,
        next_obs,
    ):
        # Guardamos la transicion en el episodio
        self.episode.append((obs, action, reward))
        
        if done:
            old_q = np.copy(self.Q)
            G = 0.0
            
            # Extraemos todos los pares (estado, accion) para agilizar la busqueda de la primera visita
            states_actions = [(s, a) for s, a, r in self.episode]
            
            # Loop for each step of episode, t = T-1, T-2, ..., 0
            for t in reversed(range(len(self.episode))):
                state, act, rew = self.episode[t]
                
                # G <- gamma * G + R_{t+1}
                G = self.discount_factor * G + rew
                
                # Unless the pair S_t, A_t appears in S_0, A_0, ..., S_{t-1}, A_{t-1}
                if (state, act) not in states_actions[:t]:
                    # Append G to Returns(S_t, A_t) y Q(S_t, A_t) <- average(Returns(S_t, A_t))
                    # Calculamos el promedio de forma incremental
                    self.n_visits[state, act] += 1.0
                    alpha = 1.0 / self.n_visits[state, act]
                    self.Q[state, act] += alpha * (G - self.Q[state, act])
                    
                    # A* <- argmax_a Q(S_t, a)
                    a_star = self._random_argmax(self.Q[state])
                    
                    # For all a in A(S_t): actualizamos la politica pi
                    num_acciones = self.env.action_space.n
                    for a in range(num_acciones):
                        if a == a_star:
                            self.pi[state, a] = 1.0 - self.epsilon + (self.epsilon / num_acciones)
                        else:
                            self.pi[state, a] = self.epsilon / num_acciones

            # Calculamos y registramos el error
            error = np.max(np.abs(self.Q - old_q))
            self.training_error.append(error)
            
            # Limpiamos el episodio
            self.episode = []