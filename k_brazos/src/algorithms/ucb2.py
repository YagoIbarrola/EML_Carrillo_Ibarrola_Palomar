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
        # self.r = np.zeros(k, dtype=int)
        # self.tau = lambda r: int(np.ceil((1 + self.alpha) ** r))

        # ka: número de épocas (rondas de selección) para cada brazo
        self.ka = np.zeros(k, dtype=int)
        
        # t: contador total de pasos (interno, para asegurar consistencia con la fórmula)
        self.total_steps = 0
        
        # Estado interno para gestionar las épocas (batches)
        self.current_batch_arm = None  # Brazo que se está ejecutando en la época actual
        self.batch_remaining = 0       # Cuántas veces falta tirar de este brazo

    def _tau(self, r: int) -> int:
        """
        Calcula la función tau(r) = ceiling((1 + alpha)^r).
        Determina el tamaño acumulado de las épocas.
        """
        return int(np.ceil((1 + self.alpha) ** r))

    # def select_arm(self) -> int:
    #     """
    #     Selecciona un brazo basado en la política UCB2.

    #     :return: índice del brazo seleccionado.
    #     """
    #     # Selecciona el brazo con la recompensa promedio estimada más alta
    #     # UCB2: Q_t(a) + sqrt((1 + alpha) * ln(e * t / tau(N_t(a))) / (2 * tau(N_t(a))))
    #     t = np.sum(self.counts)
    #     ucb_values = np.zeros(self.k)
    #     for arm in range(self.k):
    #         if self.counts[arm] == 0:
    #             return arm
    #         else:
    #             ucb_values[arm] = self.values[arm] + np.sqrt((1 + self.alpha) * np.log(np.e * t / self.tau(self.r[arm])) / (2 * self.tau(self.r[arm])))

    #     chosen_arm = np.argmax(ucb_values)
    #     self.r[chosen_arm] += 1
    #     return chosen_arm



    def select_arm(self) -> int:
            """
            Selecciona un brazo basado en UCB2.
            
            Si estamos dentro de una época (batch), devuelve el mismo brazo.
            Si no, calcula UCB2 para todos, selecciona el mejor y calcula la duración de la nueva época.
            """
            
            # 1. Fase de Inicialización (Paso 2 del pseudocódigo)
            # Selecciona cada acción una vez al principio.
            if self.total_steps < self.k:
                return self.total_steps

            # 2. Ejecución de época (Paso 5 del pseudocódigo - continuación)
            # Si todavía quedan tiradas en el bloque actual, seguimos con el mismo brazo.
            if self.batch_remaining > 0:
                self.batch_remaining -= 1
                return self.current_batch_arm

            # 3. Selección de nueva acción (Paso 4 del pseudocódigo)
            ucb_values = np.zeros(self.k)
            
            for a in range(self.k):
                # Q(a) es self.values[a]
                mean_reward = self.values[a]
                
                # Recuperamos ka (número de épocas actuales para el brazo a)
                r = self.ka[a]
                tau_r = self._tau(r)
                
                # Cálculo del término de confianza (Bonus)
                # Evitamos log(0) o división por cero, aunque la inicialización lo previene
                numerator = (1 + self.alpha) * np.log((np.e * self.total_steps) / tau_r)
                denominator = 2 * tau_r
                
                ucb_values[a] = mean_reward + np.sqrt(numerator / denominator)

            # Seleccionar la acción con el índice máximo
            selected_arm = np.argmax(ucb_values)
            
            # 4. Calcular duración de la nueva época (Paso 5 del pseudocódigo - inicio)
            # Duración = tau(ka + 1) - tau(ka)
            r_current = self.ka[selected_arm]
            duration = int(np.ceil(self._tau(r_current + 1) - self._tau(r_current)))
            
            # Actualizamos estado interno para mantener este brazo durante 'duration' pasos
            self.current_batch_arm = selected_arm
            # Restamos 1 porque vamos a devolver el brazo ahora mismo
            self.batch_remaining = max(0, duration - 1)
            
            # 5. Actualizar ka (Paso 6 del pseudocódigo)
            self.ka[selected_arm] += 1
            
            return selected_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las estadísticas.
        """
        super().update(chosen_arm, reward)
        self.total_steps += 1
