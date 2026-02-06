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


def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int):
    """
    Ejecuta experimentos comparativos entre diferentes algoritmos.
    :param bandit: Instancia de Bandit configurada para el experimento.
    :param algorithms: Lista de instancias de algoritmos a comparar.
    :param steps: Número de pasos de tiempo por ejecución.
    :param runs: Número de ejecuciones independientes.
    :return: Tuple de tres elementos: recompensas promedio, porcentaje de selecciones óptimas, y estadísticas de brazos.
    :rtype: Tuple of (np.ndarray, np.ndarray, list)
    
    Ahora devuelve también estadísticas agregadas de los brazos.
    """
    k = bandit.k
    optimal_arm = bandit.optimal_arm

    # REG 1. Obtener el valor esperado real del brazo óptimo (mu*)
    optimal_expected_value = bandit.get_expected_value(optimal_arm)

    # Inicializar matrices para recompensas y selecciones óptimas
    rewards = np.zeros((len(algorithms), steps))
    optimal_selections = np.zeros((len(algorithms), steps))
    # REG 2. Matriz para el Regret
    regret = np.zeros((len(algorithms), steps))
    # ARM 1. Inicializamos matrices para acumular estadísticas finales de los brazos (para plot_arm_statistics)
    # Acumularemos counts y values (Q) de cada 'run'
    acc_counts = np.zeros((len(algorithms), k))
    acc_values = np.zeros((len(algorithms), k))

    for run in range(runs):
        # Crear una nueva instancia del bandit para cada ejecución
        current_bandit = Bandit(arms=bandit.arms)
        
        # Obtener la recompensa esperada óptima
        # q_max = current_bandit.get_expected_value(current_bandit.optimal_arm) <------

        # Reiniciamos los algoritmos al inicio de cada run
        for algo in algorithms:
            algo.reset()

        # Inicializar recompensas acumuladas por algoritmo para esta ejecución
        # total_rewards_per_algo = np.zeros(len(algorithms))  # Para análisis por rechazo
        # Inicializar recompensas acumuladas por algoritmo para esta ejecución
        # cumulative_rewards_per_algo = np.zeros(len(algorithms))

        # Bucle de pasos (Time horizon)
        for step in range(steps):
            for idx, algo in enumerate(algorithms):
                chosen_arm = algo.select_arm()
                reward = current_bandit.pull_arm(chosen_arm)
                algo.update(chosen_arm, reward)

                rewards[idx, step] += reward
                #total_rewards_per_algo[idx] += reward

                if chosen_arm == optimal_arm:
                    optimal_selections[idx, step] += 1

                # REG 3. Calcular regret instantáneo: (Valor óptimo - Valor del elegido)
                # IMPORTANTE: Usamos .get_expected_value(), no la 'reward' con ruido

                chosen_arm_expected_value = current_bandit.get_expected_value(chosen_arm)
                regret[idx, step] += (optimal_expected_value - chosen_arm_expected_value)

        # ARM 2. Al finalizar los pasos de este 'run', acumulamos el estado final de cada algoritmo
        for idx, algo in enumerate(algorithms):
            # Asumimos que el algoritmo tiene atributos .counts y .values
            acc_counts[idx] += algo.counts 
            acc_values[idx] += algo.values

    # Promediar las recompensas y el regret sobre todas las ejecuciones
    rewards /= runs
    optimal_selections = (optimal_selections / runs) * 100
    
    # REG 4. Promediar el regret instantáneo y luego hacer la suma acumulativa
    regret /= runs
    regret_accumulated = np.cumsum(regret, axis=1) # Acumulamos el regret a lo largo del tiempo (para cada t) para cada algoritmo

    # ARM 3. Preparamos la estructura de datos para plot_arm_statistics
    # Promediamos conteos y valores acumulados
    avg_counts = acc_counts / runs
    avg_values = acc_values / runs
    
    arms_stats = []
    for idx in range(len(algorithms)):
        arms_stats.append({
            'counts': avg_counts[idx],
            'average_rewards': avg_values[idx]
        })

    return rewards, optimal_selections, arms_stats, regret_accumulated # es interesante ver regret




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