"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax (distribución de Gibbs) para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, temperature: float = 1.0, exploring: bool = False):
        """
        Inicializamos el algoritmo Softmax.

        :param k: Número de brazos.
        :param temperature: Parámetro tau (temperatura) que regula la exploración. Debe ser > 0.
        :raises ValueError: Si la temperatura es menor o igual a 0.
        """
        assert temperature > 0, "El parámetro temperatura (tau) debe ser mayor que 0."

        super().__init__(k)
        self.temperature = temperature
        self.exploring = exploring
    def select_arm(self) -> int:
        """
        Seleccionamos un brazo basado en la distribución de probabilidad de Boltzmann (Softmax).
        
        La probabilidad de elegir cada acción es proporcional al exponencial de su valor estimado
        dividido por la temperatura.

        :return: Índice del brazo seleccionado.
        """
        
        # Obtenemos los valores Q actuales (self.values)
        # Aplicamos la fórmula: e^(Q(a)/tau)
        # Nota: Para estabilidad numérica, a veces se resta el max(Q) antes de exponenciar, 
        if self.exploring:
            # Hacemos una primera exploración inicial de todos los brazos (solo en la primera llamada a select_arm)
            for arm in range(self.k):
                if self.counts[arm] == 0:  # Si el brazo no ha sido seleccionado aún
                    return arm  # Seleccionamos ese brazo para explorarlo
                
        exp_values = np.exp(self.values / self.temperature)
        
        # Calculamos el denominador: suma de todos los exponenciales
        total_sum = np.sum(exp_values)
        
        # Calculamos las probabilidades pi_t(a)
        probabilities = exp_values / total_sum
        
        # Seleccionamos un brazo aleatorio basándonos en las probabilidades calculadas
        chosen_arm = np.random.choice(self.k, p=probabilities)

        return chosen_arm