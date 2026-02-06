"""
Module: algorithms/ucb2.py
Description: Implementación del algoritmo UCB2 para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB2(Algorithm):

    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.
        :param k: Número de brazos.
        :param alpha: Parámetro de exploración (0 < alpha < 1).
        :raises ValueError: Si alpha no está en (0, 1).
        """
        assert 0 < alpha < 1, "El parámetro alpha debe estar entre 0 y 1."

        super().__init__(k)
        self.alpha = alpha

        # ka: número de épocas (rondas de selección) para cada brazo
        self.ka = np.zeros(k, dtype=int)
        
        # Estado interno para gestionar las épocas (batches)
        self.current_batch_arm = None  # Brazo que se está ejecutando en la época actual
        self.batch_remaining = 0       # Cuántas veces falta tirar de este brazo

    # BASÁNDONOS en epsilon_greedy.py, el pseudocódigo de la diapositiva 38 y la fórmula de la 38

    def _tau(self, r: int) -> int:
        """
        Calcula la función tau(ka) = ceiling((1 + alpha)^r). Siendo r = ka
        Determina el tamaño acumulado de las épocas.
        """
        return int(np.ceil((1 + self.alpha) ** r))

    def select_arm(self) -> int:
            """
            Selecciona un brazo basado en UCB2.
            
            Si estamos dentro de una época (batch), devuelve el mismo brazo.
            Si no, calcula UCB2 para todos, selecciona el mejor y calcula la duración de la nueva época.
            """
            
            # Selecciona cada acción una vez al principio.
            t = np.sum(self.counts)
            if t < self.k: # Selecciona cada acción una vez al principio.
                return t

            # Si todavía quedan tiradas en el bloque actual, seguimos con el mismo brazo.
            if self.batch_remaining > 0:
                self.batch_remaining -= 1
                return self.current_batch_arm

            # Selección de nueva acción
            ucb_values = np.zeros(self.k)
            
            for a in range(self.k):
                # Q(a) es self.values[a]
                mean_reward = self.values[a]
                
                # Recuperamos ka (número de épocas actuales para el brazo a)
                r = self.ka[a]
                tau_ka = self._tau(r)
                
                numerator = (1 + self.alpha) * np.log((np.e * t) / tau_ka)
                denominator = 2 * tau_ka
                
                ucb_values[a] = mean_reward + np.sqrt(numerator / denominator)

            # Seleccionar la acción con el índice máximo
            selected_arm = np.argmax(ucb_values)
            
            # Calcular duración de la nueva época. Duración = tau(ka + 1) - tau(ka)
            ka_current = self.ka[selected_arm]
            duration = int(np.ceil(self._tau(ka_current + 1) - self._tau(ka_current)))
            
            # Actualizamos estado interno para mantener este brazo durante 'duration' pasos
            self.current_batch_arm = selected_arm
            # Restamos 1 porque vamos a devolver el brazo ahora mismo
            self.batch_remaining = max(0, duration - 1)
            
            # Actualizar ka
            self.ka[selected_arm] += 1
            
            return selected_arm

    # def update(self, chosen_arm: int, reward: float):
    #     """
    #     Actualiza las estadísticas.
    #     """
    #     super().update(chosen_arm, reward)
    #     self.total_steps += 1
    #     print(f"UCB2 Update: chosen_arm={chosen_arm}, reward={reward}, total_steps={self.total_steps}, ka={self.ka}, batch_remaining={self.batch_remaining}")
