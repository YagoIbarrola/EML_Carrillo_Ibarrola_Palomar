import numpy as np
import matplotlib.pyplot as plt

def graficar_entrenamiento(env, agent, rolling_length=500):
    # Definimos una función interna para calcular las medias móviles
    def get_moving_avgs(arr, window, convolution_mode):
        # Calculamos la media móvil para suavizar los datos ruidosos
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Creamos una figura con tres subgráficos alineados horizontalmente
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Configuramos y graficamos las recompensas por episodio
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Configuramos y graficamos las longitudes de los episodios
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Configuramos y graficamos el error de entrenamiento
    axs[2].set_title("Training Evolution")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference")
    # Eje Y logarítmico para visualizar mejor la evolución del error a lo largo del entrenamiento
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Step")

    # Ajustamos el espaciado y mostramos la figura en pantalla
    plt.tight_layout()
    plt.show()