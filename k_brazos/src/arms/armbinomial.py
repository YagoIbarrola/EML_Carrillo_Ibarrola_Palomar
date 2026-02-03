"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the Binomial distribution arm.
"""

import numpy as np
from arms import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializamos el brazo con distribución binomial.

        :param n: Número de ensayos (debe ser entero positivo).
        :param p: Probabilidad de éxito en cada ensayo (entre 0 y 1).
        """
        # Verificamos que los parámetros sean válidos
        assert isinstance(n, int) and n > 0, "El parámetro n debe ser un entero positivo."
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Generamos una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        # Obtenemos una muestra aleatoria de la distribución binomial
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devolvemos el valor esperado de la distribución binomial.

        :return: Valor esperado (n * p).
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo.
        """
        return f"ArmBinomial(n={self.n}, p={self.p:.2f})"

    @classmethod
    def generate_arms(cls, k: int, n: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Generamos k brazos con probabilidades únicas p en el rango [p_min, p_max]
        y un n fijo para todos.

        :param k: Número de brazos a generar.
        :param n: Número de ensayos fijo para todos los brazos.
        :param p_min: Probabilidad mínima.
        :param p_max: Probabilidad máxima.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "El rango de probabilidades [p_min, p_max] no es válido."

        # Generamos k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 4)  # Redondeamos para facilitar la unicidad y lectura
            p_values.add(p)

        p_values = list(p_values)
        
        # Creamos la lista de brazos con el n fijo y los p variables
        arms = [cls(n, p) for p in p_values]

        return arms