import numpy as np
import gymnasium as gym
import imageio
from IPython.display import Image, display

def animar_estados_taxi_gif(secuencia_estados, nombre_archivo="secuencia_taxi.gif", fps=4):
    # Inicializamos el entorno en modo grafico para obtener imagenes
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    env.reset()
    
    # Preparamos una lista vacia donde guardaremos los fotogramas
    fotogramas = []
    
    # Recorremos cada par (estado, is_exploring)
    for estado, is_exploring in secuencia_estados:
        # Modificamos el estado interno del entorno
        env.unwrapped.s = estado
        
        # Capturamos la imagen original (es un array de NumPy RGB)
        frame = env.render()
        
        # Si el agente estaba explorando, aplicamos el efecto visual
        if is_exploring:
            # Hacemos una copia para no alterar lecturas internas
            frame = np.copy(frame)
            
            # Efecto: Marco rojo de 10 píxeles de grosor
            grosor = 10
            color_rojo = [255, 0, 0] # R, G, B
            
            # Bordes horizontales (arriba y abajo)
            frame[:grosor, :] = color_rojo
            frame[-grosor:, :] = color_rojo
            
            # Bordes verticales (izquierda y derecha)
            frame[:, :grosor] = color_rojo
            frame[:, -grosor:] = color_rojo
            
            frame_float = frame.astype(np.float32)
            frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] + 80, 0, 255) # Aumentar rojo
            frame = frame_float.astype(np.uint8)

        # Anadimos a nuestra lista
        fotogramas.append(frame)
        
    # Cerramos el entorno para liberar los recursos
    env.close()

    # Añadimos un par de fotogramas negros al final para marcar el fin de la secuencia
    # (Tomamos el tamaño del primer fotograma como referencia)
    frame_negro = np.zeros_like(fotogramas[0])
    fotogramas.append(frame_negro)
    fotogramas.append(frame_negro)

    # Generamos el archivo GIF uniendo todos los fotogramas
    imageio.mimsave(nombre_archivo, fotogramas, fps=fps, loop=0)
    
    # Mostramos el GIF resultante directamente en el notebook
    display(Image(filename=nombre_archivo))