import numpy as np
import gymnasium as gym
import imageio
from IPython.display import Image as IPyImage, display
from PIL import Image as PILImage  # Importamos Pillow para manejar las superposiciones

ICON_FILES = {
        0 : "down.png",
        1 : "up.png",
        2 : "right.png",
        3 : "left.png",
        4 : "pick_up.png",
        5 : "drop_off.png"
    }

def animar_estados_taxi_gif(
    episode_history, 
    nombre_archivo="secuencia_taxi.gif", 
    fps=4
):
    
    # 1. Cargar y preparar los iconos en memoria antes del bucle
    # No sabemos el tamaño exacto aún, lo redimensionaremos dinámicamente
    iconos_cargados = {}
    for accion_id, filename in ICON_FILES.items():
        # Convertimos a RGBA para mantener el fondo transparente del PNG
        iconos_cargados[accion_id] = PILImage.open("../data/icons_taxi" + filename).convert("RGBA")

    # Inicializamos el entorno en modo grafico para obtener imagenes
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    env.reset()
    
    # Preparamos una lista vacia donde guardaremos los fotogramas
    fotogramas = []
    
    # Recorremos cada par (estado, is_exploring)
    for estado, is_exploring, tabla_acciones in episode_history:
        # Modificamos el estado interno del entorno
        env.unwrapped.s = estado
        
        # Capturamos la imagen original (es un array de NumPy RGB)
        frame = env.render()
        
        # Si el agente estaba explorando, aplicamos el efecto visual
        if is_exploring:
            frame = np.copy(frame)
            grosor = 10
            color_rojo = [255, 0, 0] # R, G, B
            
            # Bordes
            frame[:grosor, :] = color_rojo
            frame[-grosor:, :] = color_rojo
            frame[:, :grosor] = color_rojo
            frame[:, -grosor:] = color_rojo
            
            # Tinte rojizo
            frame_float = frame.astype(np.float32)
            frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] + 80, 0, 255)
            frame = frame_float.astype(np.uint8)

        # --- NUEVA LÓGICA DE SUPERPOSICIÓN DE ICONOS ---
        
        # Calculamos el tamaño de cada celda basándonos en el tamaño del frame (Gymnasium Taxi es 5x5 celdas)
        alto_frame, ancho_frame, _ = frame.shape
        ancho_celda = ancho_frame // 11
        alto_celda = alto_frame // 7
        
        # Convertimos el frame de NumPy a una imagen de Pillow para facilitar el pegado con transparencia
        img_frame = PILImage.fromarray(frame).convert("RGBA")
        
        # Recorremos la tabla de valores (asumimos que es una matriz de 5x5)
        for fila_idx, fila in enumerate(range(1, 6)):
            for col_idx, col in enumerate(range(1, 11, 2)):
                accion = tabla_acciones[fila_idx][col_idx]
                
                if accion in iconos_cargados:
                    icono = iconos_cargados[accion]
                    
                    # Redimensionamos el icono para que quepa en la celda (ej. 50% del tamaño de la celda)
                    tamano_icono = (int(ancho_celda * 0.5), int(alto_celda * 0.5))
                    icono_redimensionado = icono.resize(tamano_icono, PILImage.Resampling.LANCZOS)
                    
                    # Calculamos las coordenadas X e Y para centrar el icono en la celda actual
                    x = (col * ancho_celda) + (ancho_celda - tamano_icono[0]) // 2
                    y = (fila * alto_celda) + (alto_celda - tamano_icono[1]) // 2
                    
                    # Pegamos el icono sobre el frame. 
                    # El 3er argumento actúa como "máscara" para que respete la transparencia del PNG.
                    img_frame.paste(icono_redimensionado, (x, y+10), icono_redimensionado)
        
        # Convertimos de vuelta a RGB de NumPy y lo añadimos a la lista
        frame_final = np.array(img_frame.convert("RGB"))
        fotogramas.append(frame_final)
        
    # Cerramos el entorno para liberar los recursos
    env.close()

    # Añadimos fotogramas negros al final
    frame_negro = np.zeros_like(fotogramas[0])
    fotogramas.append(frame_negro)
    fotogramas.append(frame_negro)

    # Generamos el archivo GIF
    imageio.mimsave(nombre_archivo, fotogramas, fps=fps, loop=0)
    
    # Mostramos el GIF resultante
    display(IPyImage(filename=nombre_archivo))