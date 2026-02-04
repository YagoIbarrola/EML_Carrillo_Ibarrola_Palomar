"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (c > 0).
        :raises ValueError: Si c no es positivo.
        """
        assert c > 0, "El parámetro c debe ser positivo."

        super().__init__(k)
        self.c = c

    # BASÁNDONOS en epsilon_greedy.py y la fórmula de la 37
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.

        :return: índice del brazo seleccionado.
        """
        # Selecciona el brazo con la recompensa promedio estimada más alta
        # UCB1: Q_t(a) + c * sqrt(ln(t) / N_t(a))
        t = np.sum(self.counts)
        if t == 0:
            chosen_arm = np.random.choice(self.k)
        else:
            ucb_values = self.values + self.c * np.sqrt(np.log(t) / (self.counts + 1e-8))  # Evitar división por cero
            chosen_arm = np.argmax(ucb_values)

        return chosen_arm




