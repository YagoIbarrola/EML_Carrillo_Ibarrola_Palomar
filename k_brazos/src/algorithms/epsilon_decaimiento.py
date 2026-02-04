"""
Module: algorithms/epsilon_decaimiento.py
Description: Implementación del algoritmo epsilon-decaimiento.
             Reduce la exploración a medida que pasa el tiempo.
"""

import numpy as np

from algorithms.algorithm import Algorithm

class EpsilonDecaimiento(Algorithm):

    def __init__(self, k: int, epsilon_0: float = 1.0, lambda_decay: float = 0.01, epsilon_min: float = 0.01):
        """
        Inicializa el algoritmo epsilon-decaimiento.

        :param k: Número de brazos.
        :param epsilon_0: Valor inicial de exploración (por defecto 1.0).
        :param lambda_decay: Tasa de decaimiento (por defecto 0.01).
        :param epsilon_min: Valor mínimo de exploración (por defecto 0.01).
        """
        assert 0 <= epsilon_0 <= 1, "El parámetro epsilon_0 debe estar entre 0 y 1."
        assert 0 <= epsilon_min <= 1, "El parámetro epsilon_min debe estar entre 0 y 1."
        assert epsilon_min <= epsilon_0, "epsilon_min debe ser menor o igual que epsilon_0."

        super().__init__(k)
        self.epsilon_0 = epsilon_0
        self.lambda_decay = lambda_decay
        self.epsilon_min = epsilon_min

    # BASÁNDONOS en epsilon_greedy.py, el pseudocódigo de la diapositiva 28 y la fórmula de la 29
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-decaimiento.
        
        Si hay brazos sin explorar, se seleccionan primero de ellos.
        Calcula epsilon_t basado en la fórmula de decaimiento inversamente proporcional.
        Selecciona entre exploración (azar) o explotación (greedy) según epsilon_t.

        :return: Índice del brazo seleccionado.
        """

        # Generar cada opción k una vez para inicializar
        # Primero comprueba si queda algún brazo sin explorar.
        for arm_index in range(self.k):
            if self.counts[arm_index] == 0:
                return arm_index

        # Calcular epsilon_t
        # t es el número total de pasos dados hasta ahora
        t = np.sum(self.counts)

        # Fórmula de decaimiento inversamente proporcional: f(t) = e0 / (1 + lambda * t)
        decayed_epsilon = self.epsilon_0 / (1 + self.lambda_decay * t)
        
        # Aplicamos el máximo con epsilon_min
        current_epsilon = max(self.epsilon_min, decayed_epsilon)

        # Selección con probabilidad epsilon_t
        if np.random.random() < current_epsilon:
            # Selecciona un brazo al azar (Exploración)
            chosen_arm = np.random.choice(self.k)
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta (Explotación)
            # Nota: np.argmax devuelve el primer índice en caso de empate.
            chosen_arm = np.argmax(self.values)

        return chosen_arm