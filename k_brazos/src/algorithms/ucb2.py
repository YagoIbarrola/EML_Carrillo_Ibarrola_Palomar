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
        self.k = k
        # Número de épocas para cada brazo
        self.num_epoch_arm = np.zeros(self.k, dtype=int)
        
        # Estado interno para gestionar las épocas
        self.current_batch_arm = None  # Brazo que se está ejecutando en la época actual
        self.batch_remaining = 0       # Cuántas veces falta tirar de este brazo

    # BASÁNDONOS en epsilon_greedy.py, el pseudocódigo de la diapositiva 38 y la fórmula de la 38

    def _tau(self, r: int):
        """
        Calcula la función tau(r) = (1 + alpha)^r.
        Determina el tamaño acumulado de las épocas.
        """
        return (1 + self.alpha) ** r

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
                
                # Obtenemos tau del número de épocas actuales para el brazo a
                tau_r = self._tau(self.num_epoch_arm[a])

                numerator = (1 + self.alpha) * np.log((np.e * t) / tau_r)
                denominator = 2 * tau_r
                
                ucb_values[a] = mean_reward + np.sqrt(numerator / denominator)

            # Seleccionar la acción con el índice máximo
            selected_arm = np.argmax(ucb_values)
            
            # Calcular duración de la nueva época. Duración = tau(r + 1) - tau(r)
            r = self.num_epoch_arm[selected_arm]
            duration = int(np.ceil(self._tau(r + 1) - self._tau(r)))
            
            # Actualizamos estado interno para mantener este brazo durante 'duration' pasos
            self.current_batch_arm = selected_arm
            # Restamos 1 porque vamos a devolver el brazo ahora mismo
            self.batch_remaining = max(0, duration - 1)
            
            # Actualizar num_epoch_arm
            self.num_epoch_arm[selected_arm] += 1
            
            return selected_arm

    def reset(self):
        """
        Reinicia el algoritmo.
        """
        super().reset()
        # Número de épocas para cada brazo
        self.num_epoch_arm = np.zeros(self.k, dtype=int)
        
        # Estado interno para gestionar las épocas
        self.current_batch_arm = None  # Brazo que se está ejecutando en la época actual
        self.batch_remaining = 0       # Cuántas veces falta tirar de este brazo
