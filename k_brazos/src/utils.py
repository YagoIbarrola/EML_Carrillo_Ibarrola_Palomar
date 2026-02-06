from typing import List

import numpy as np

from algorithms import Algorithm

from arms import Bandit

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
            acc_values[idx] += algo.values * algo.counts

    # Promediar las recompensas y el regret sobre todas las ejecuciones
    rewards /= runs
    optimal_selections = (optimal_selections / runs) * 100
    
    # REG 4. Promediar el regret instantáneo y luego hacer la suma acumulativa
    regret /= runs
    regret_accumulated = np.cumsum(regret, axis=1) # Acumulamos el regret a lo largo del tiempo (para cada t) para cada algoritmo

    # ARM 3. Preparamos la estructura de datos para plot_arm_statistics
    # Promediamos conteos y valores acumulados
    avg_values = acc_values / acc_counts
    avg_counts = acc_counts / runs
    
    
    arms_stats = []
    for idx in range(len(algorithms)):
        arms_stats.append({
            'counts': avg_counts[idx],
            'average_rewards': avg_values[idx]
        })

    return rewards, optimal_selections, arms_stats, regret_accumulated # es interesante ver regret