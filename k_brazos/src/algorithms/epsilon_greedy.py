"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class EpsilonGreedy(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy.
        Asegura que todos los brazos se prueben al menos una vez antes de explotar.

        :return: índice del brazo seleccionado.
        """

 # Observa que para para epsilon=0 solo selecciona un brazo y no hace un primer recorrido por todos ellos. -----> Con epsilon=0.1 tampoco
 # ¿Podrías modificar el código para que funcione correctamente para epsilon=0?

        # Primero comprueba si queda algún brazo sin explorar.
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm
        
        #-----> ALTERNATIVA Seleccionar brazos no visitados
        # unplayed_arms = np.where(self.counts == 0)[0]
        # if len(unplayed_arms) > 0:
        #     return np.random.choice(unplayed_arms)
            
        # Una vez que todos los brazos tienen al menos una muestra, aplica la lógica estándar
        if np.random.random() < self.epsilon:
            # Selecciona un brazo al azar (Exploración)
            chosen_arm = np.random.choice(self.k)
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta (Explotación)
            # Nota: np.argmax devuelve el primer índice en caso de empate.
            chosen_arm = np.argmax(self.values)

        return chosen_arm