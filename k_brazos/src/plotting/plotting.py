"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, EpsilonDecaimiento, UCB1, UCB2, Softmax


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # Añadimos casos para otros algoritmos
    elif isinstance(algo, EpsilonDecaimiento):
        label += f" (epsilon_0={algo.epsilon_0}, decaimiento={algo.lambda_decay}, epsilon_min={algo.epsilon_min})"
    elif isinstance(algo, UCB1):
        label += f" (C={algo.c})"
    elif isinstance(algo, UCB2):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, Softmax):
        label += f" (temperatura={algo.temperature})"
        
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2, alpha=0.8)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

# Nuevo
def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de porcentaje de selección del brazo óptimo vs pasos de tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        # Damos diferentes formas a las líneas según el algoritmo para mejorar la distinción visual
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2, alpha=0.8, linestyle='-')

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Porcentaje de selección del brazo óptimo (%)', fontsize=14)
    plt.title('Porcentaje de selección del brazo óptimo vs pasos de tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_arm_statistics(arms_stats: list, algorithms: List[Algorithm], optimal_arm_index: int = None):
    """
    Genera gráficas de estadísticas por brazo para cada algoritmo.
    Muestra el promedio de recompensas por brazo y cuántas veces fue seleccionado.
    """
    
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    
    n_algos = len(algorithms)
    
    # Mantenemos una altura razonable por subplot
    fig, axes = plt.subplots(n_algos, 1, figsize=(12, 6 * n_algos), sharex=False)
    
    if n_algos == 1:
        axes = [axes]
        
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        stats = arms_stats[idx]
        
        counts = stats.get('counts')
        avg_rewards = stats.get('average_rewards')
        k_arms = len(counts)
        
        colors = ['#4c72b0'] * k_arms
        if optimal_arm_index is not None and 0 <= optimal_arm_index < k_arms:
            colors[optimal_arm_index] = '#55a868'
            
        bars = ax.bar(range(1,k_arms+1), avg_rewards, color=colors, alpha=0.8)
        
        algo_name = get_algorithm_label(algo)
        ax.set_title(f'Estadísticas por Brazo: {algo_name}', fontsize=16, pad=15) 
        ax.set_ylabel('Recompensa Promedio Estimada', fontsize=12)
        ax.set_xlabel('Índice del Brazo', fontsize=12)
        ax.set_xticks(range(1,k_arms+1))
        
        # Ajustamos límite Y para las etiquetas
        max_height = max(avg_rewards) if len(avg_rewards) > 0 else 1.0
        ax.set_ylim(0, max_height * 1.25) 

        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = int(counts[i])
            
            label_text = f"N={count}"
            if i == optimal_arm_index:
                label_text += "\n(Opt)"
                
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label_text,
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    # Primero aplicamos un ajuste automático básico
    plt.tight_layout()
    
    # --- CORRECCIÓN: Ajuste manual del espacio vertical ---
    # hspace controla el espacio entre subplots (como fracción de la altura media de los ejes)
    # Un valor de 0.5 suele ser suficiente para evitar solapamientos con títulos grandes
    plt.subplots_adjust(hspace=0.5)
    
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    El regret se define como la diferencia entre la recompensa óptima esperada y la obtenida.

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Argumentos extra.
    """
    plt.figure(figsize=(14, 7))

    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2, alpha=0.8)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret acumulado', fontsize=14)
    plt.title('Evolución del regret acumulado', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()