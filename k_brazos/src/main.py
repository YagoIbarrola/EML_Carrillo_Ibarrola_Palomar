"""
Module: main.py
Description: Main script to run comparative experiments between different algorithms.
El experimento compara el rendimiento de algoritmos epsilon-greedy en un problema de k-armed bandit.    
Se generan gráficas de recompensas promedio y selecciones óptimas para cada algoritmo.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np

from algorithms import Algorithm, EpsilonGreedy, EpsilonDecaimiento, UCB1, UCB2, Softmax
from arms import ArmNormal, ArmBernoulli, ArmBinomial, Bandit
from plotting import plot_average_rewards, plot_optimal_selections, plot_arm_statistics, plot_regret
from utils import run_experiment


def main():
    """
    Main function to set up and execute comparative experiments.
    """
    seed = 42
    np.random.seed(seed)

    k = 10  # Número de brazos
    steps = 1000  # Número de pasos
    runs = 500  # Número de ejecuciones

    #bandit = Bandit(arms=ArmNormal.generate_arms(k))
    # bandit = Bandit(arms=ArmBernoulli.generate_arms(k, scale = 10))  
    bandit = Bandit(arms=ArmBinomial.generate_arms(k, n=10, scale = 1))
    print(bandit)

    # Obtenemos el brazo óptimo para pasarlo a la gráfica
    optimal_arm_index = bandit.optimal_arm
    print(f"Optimal arm: {optimal_arm_index + 1} with expected reward={bandit.get_expected_value(optimal_arm_index)}")

    # Añadidos nuevos algoritmos para comparar
    algorithms = [
                    EpsilonGreedy(k=k, epsilon=0), 
                    EpsilonGreedy(k=k, epsilon=0.01), 
                    EpsilonGreedy(k=k, epsilon=0.1), 
                    #EpsilonDecaimiento(k=k, epsilon_0=1.0, lambda_decay=0.01, epsilon_min=0.01), 
                    #UCB1(k=k, c=1.0), 
                    #UCB2(k=k, alpha=0.1), 
                    #Softmax(k=k, temperature=0.1), 
                    #Softmax(k=k, temperature=1.0)
                ]

    # Ejecutar el experimento y obtener las recompensas promedio y selecciones óptimas
    rewards, optimal_selections, arms_stats, regret_accumulated = run_experiment(bandit, algorithms, steps, runs)

    # Generar las gráficas utilizando las funciones externas
    # Añadimos nuevas gráficas
    plot_average_rewards(steps, rewards, algorithms)
    plot_optimal_selections(steps, optimal_selections, algorithms)
    plot_arm_statistics(arms_stats, algorithms, optimal_arm_index)
    plot_regret(steps, regret_accumulated, algorithms)

if __name__ == '__main__':
    main()