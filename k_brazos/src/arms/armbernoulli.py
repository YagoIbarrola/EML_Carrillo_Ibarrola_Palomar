"""
Module: arms/armbernoulli.py
Description: Contains the implementation of the ArmBernoulli class.
"""

import numpy as np

from arms.armbinomial import ArmBinomial

class ArmBernoulli(ArmBinomial):
    def __init__(self, p: float, scale: float = 1.0):
        """
        Inicializamos el brazo de Bernoulli como un caso concreto de Binomial con n=1.

        :param p: Probabilidad de éxito (entre 0 y 1).
        """
        # Llamamos al constructor de la clase padre fijando n=1
        super().__init__(n=1, p=p, scale=scale)

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción del brazo con su probabilidad p.
        """
        return f"ArmBernoulli(p={self.p:.2f})"

    # Nota: No necesitamos reescribir pull() ni get_expected_value()
    # porque la lógica de ArmBinomial ya funciona correctamente para n=1.

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 1.0, scale: float = 1.0):
        """
        Generamos k brazos de Bernoulli con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Probabilidad mínima.
        :param p_max: Probabilidad máxima.
        :return: Lista de brazos Bernoulli generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "El rango de probabilidades [p_min, p_max] no es válido."

        # Generamos k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 4)
            p_values.add(p)

        p_values = list(p_values)

        # Creamos la lista de brazos instanciando la clase actual (ArmBernoulli)
        arms = [cls(p, scale = scale) for p in p_values]

        return arms