import gymnasium as gym
import imageio
import numpy as np
from IPython.display import Image, display

def animar_estados_taxi_gif(secuencia_estados, nombre_archivo="secuencia_taxi.gif", fps=4):
    # Inicializamos el entorno en modo grafico para obtener imagenes
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    env.reset()
    
    # Preparamos una lista vacia donde guardaremos los fotogramas
    fotogramas = []
    
    # Recorremos cada estado numerico de la lista recibida
    for estado in secuencia_estados:
        # Modificamos el estado interno del entorno
        env.unwrapped.s = estado
        
        # Capturamos la imagen resultante y la anadimos a nuestra lista
        fotogramas.append(env.render())
        
    # Cerramos el entorno para liberar los recursos
    env.close()

    # Añadimos un par de fotogramas negros al final para marcar el fin de la secuencia
    
    fotogramas.append(np.zeros_like(fotogramas[0]))
    fotogramas.append(np.zeros_like(fotogramas[0]))

    # Generamos el archivo GIF uniendo todos los fotogramas
    imageio.mimsave(nombre_archivo, fotogramas, fps=fps, loop=0)
    
    # Mostramos el GIF resultante directamente en el notebook
    display(Image(filename=nombre_archivo))