import numpy as np
from matplotlib import pyplot as plt


def get_moving_avgs(arr, window: int, convolution_mode: str = "valid"):
    """
    Compute moving average using convolution.

    Args:
        arr: Sequence of numeric values
        window: Window size
        convolution_mode: 'valid', 'same', or 'full'

    Returns:
        Smoothed numpy array
    """
    return (
        np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode,
        )
        / window
    )


def plot_training_metrics(
    rewards,
    lengths,
    training_errors,
    rolling_length: int = 500,
    isMonteCarlo: bool = False,
):
    """
    Plot smoothed training statistics.

    Args:
        rewards: Episode reward history
        lengths: Episode length history
        training_errors: TD error history
        rolling_length: Moving average window
    """

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))


    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        rewards,
        rolling_length,
        "valid",
    )
    axs[0].plot(reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        lengths,
        rolling_length,
        "valid",
    )
    axs[1].plot(length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    if isMonteCarlo:
        title = "Training Evolution"
        ylabel = "Temporal Difference"
        axs[2].set_yscale("log")
    else:
        title = "Delta_Q"
        ylabel = "Temporal Difference Error"
    axs[2].set_title(title)
    training_error_moving_average = get_moving_avgs(
        training_errors,
        rolling_length,
        "same",
    )
    axs[2].plot(training_error_moving_average)
    axs[2].set_ylabel(ylabel)
    axs[2].set_xlabel("Step")

    for i, ax in enumerate(axs):
        ax.tick_params(axis='x', labelrotation=30)

    plt.tight_layout()
    plt.show()



def save_state(agent, is_exploring, obs, episode_history):
    policy = agent.get_current_policy()
    passenger_loc = (obs // 4) % 5  # Extraemos la ubicación del pasajero del estado
    destination_idx = obs % 4           # Extraemos el destino del estado
    tabla_acciones = get_policy_grid(agent.env, policy, passenger_loc, destination_idx)
    episode_history.append((obs, is_exploring, tabla_acciones))

def get_policy_grid(env, policy, passenger_loc, destination_idx):
    """
    Extrae una cuadrícula de 5x5 con las acciones óptimas asumiendo una 
    ubicación fija para el pasajero y el destino.
    """
    grid = np.zeros((5, 5), dtype=int)
    
    for row in range(5):
        for col in range(5):
            # Calculamos el número de estado (0-499) para esta combinación exacta
            state = env.unwrapped.encode(row, col, passenger_loc, destination_idx)
            
            # Guardamos la acción preferida en nuestra cuadrícula
            grid[row][col] = policy[state]
            
    return grid

def save_training_metrics(rewards, lengths, training_errors, filename):
    """
    Guarda las métricas de entrenamiento en un archivo Numpy (.npz).

    Args:
        rewards: Lista de recompensas por episodio
        lengths: Lista de longitudes de episodio
        training_errors: Lista de errores de entrenamiento (TD error)
        filename: Nombre del archivo Numpy (.npz) para guardar las métricas
    """
    np.savez(filename, rewards=rewards, lengths=lengths, training_errors=training_errors)

def load_training_metrics(filename):
    """
    Carga las métricas de entrenamiento desde un archivo CSV.

    Args:
        filename: Nombre del archivo CSV que contiene las métricas

    Returns:
        Tuple of lists: (rewards, lengths, training_errors)
    """
    data = np.load(filename)
    rewards = data['rewards']
    lengths = data['lengths']
    training_errors = data['training_errors']

    return rewards, lengths, training_errors