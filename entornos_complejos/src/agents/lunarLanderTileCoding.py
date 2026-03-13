import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
import random

class TileCodingEnv(ObservationWrapper):
    """
    TileCodingEnv es un envoltorio para un entorno Gym que aplica la técnica de Tile Coding.
    Esta técnica discretiza observaciones continuas en múltiples rejillas (tilings) desplazadas,
    permitiendo representar el espacio de estados de forma que se faciliten la generalización y el aprendizaje.
    """

    def __init__(self, env, bins, low, high, n_tilings=4, passthrough_dims=None):
        """
        Inicializa el entorno env con tile coding.

        Parámetros:
        - env: entorno original de Gym.
        - bins: array o lista con el número de intervalos (bins) que hay que particionar cada dimensión.
        - low: array con el límite inferior para cada dimensión.
        - high: array con el límite superior para cada dimensión.
        - n_tilings: número de tilings (rejillas) a crear (por defecto 4).
        - passthrough_dims: dimensiones que no se discretizan (se mantienen como índices enteros).

        Se llama al método _create_tilings para generar las rejillas desplazadas.
        """
        super().__init__(env)  # Llama al constructor de la clase padre ObservationWrapper.

        bins = np.asarray(bins, dtype=np.int64)
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)

        if bins.ndim != 1 or low.ndim != 1 or high.ndim != 1:
            raise ValueError("bins, low y high deben ser vectores 1D")
        if not (len(bins) == len(low) == len(high)):
            raise ValueError("bins, low y high deben tener la misma longitud")
        if np.any(bins < 2):
            raise ValueError("Cada dimensión en bins debe ser >= 2")
        if n_tilings < 1:
            raise ValueError("n_tilings debe ser >= 1")
        if np.any(high <= low):
            raise ValueError("Cada componente de high debe ser mayor que low")

        if passthrough_dims is None:
            # Por defecto, tratamos como passthrough las dimensiones binarias (ej. patas de LunarLander).
            passthrough_mask = (bins == 2) & np.isclose(low, 0.0) & np.isclose(high, 1.0)
        else:
            passthrough_mask = np.zeros(len(bins), dtype=bool)
            passthrough_mask[np.asarray(passthrough_dims, dtype=np.int64)] = True

        # Guardamos atributos útiles para calcular las features solo una vez.

        self.tilings = self._create_tilings(
            bins=bins,
            high=high,
            low=low,
            n_tilings=n_tilings,
            passthrough_mask=passthrough_mask,
        )
        self.n_tilings = n_tilings
        self.bins = bins
        self.low = low
        self.high = high
        self.passthrough_mask = passthrough_mask
        # el vector de observación tendrá C componentes. Por ejemplo, para 2 dimensiones × 4 tilings = C = 8.
        self.observation_space = gym.spaces.MultiDiscrete(nvec=(bins.tolist() * n_tilings))


    def observation(self, obs):  # Es necesario sobreescribir este método de ObservationWrapper
        """
        Transforma una observación continua en una representación discreta usando tile coding.

        Parámetro:
        - obs: observación continua proveniente del entorno.

        Para cada tiling en self.tilings, se determina el índice (bin) para cada dimensión usando np.digitize.
        Se devuelve una lista de tuplas de índices, una por cada tiling.
        Antes de retornar, se calcula y almacena en self.last_active_features el conjunto de índices
        activos (flattened) usando _get_active_features().

        Retorna:
        - indices: lista de tuplas de índices, una por cada tiling.

        """
        # indices = []  # Lista que almacenará los índices discretizados para cada tiling.
        # for t in self.tilings:
        #     # Para cada tiling 't', se calcula el índice en el que se encuentra cada componente de la observación.
        #     tiling_indices = tuple(np.digitize(i, b) for i, b in zip(obs, t))
        #     indices.append(tiling_indices)  # Se agrega la tupla de índices correspondiente a la tiling actual.

        # # Calcula y guarda las features activas a partir de los índices obtenidos.
        # self.last_active_features = self._get_active_features(indices)
        # return indices # Retorna la lista de índices de todas las tilings.
        indices = []
        for tiling in self.tilings:
            tiling_indices = []
            for dim, value in enumerate(obs):
                if self.passthrough_mask[dim]:
                    idx = int(np.clip(np.rint(value), 0, self.bins[dim] - 1))
                else:
                    idx = int(np.digitize(value, tiling[dim]))
                tiling_indices.append(idx)
            indices.append(tuple(tiling_indices))
        return indices

        
    

    def _create_tilings(self, bins, high, low, n_tilings, passthrough_mask):
        """
        Crea 'n' tilings (rejillas) desplazadas para el tile coding.

        Parámetros:
        - bins: número de intervalos (bins) en cada dimensión.
        - high: array con el límite superior para cada dimensión.
        - low: array con el límite inferior para cada dimensión.
        - n: número de tilings a crear.

        El proceso consiste en:
         1. Generar un vector de desplazamientos base (displacement_vector) para cada dimensión.
         2. Para cada tiling, se ajustan los límites 'low' y 'high' añadiéndoles un pequeño desplazamiento aleatorio.
         3. Se calculan los tamaños de los segmentos en cada dimensión (segment_sizes).
         4. Se determinan desplazamientos específicos para cada dimensión y se aplican a los límites.
         5. Finalmente, se generan los buckets (límites discretos) para cada dimensión usando np.linspace.

        Retorna:
        - tilings: una lista donde cada elemento es una tiling (lista de arrays de buckets para cada dimensión).
        """
        # Se genera un vector de desplazamientos en cada dimensión en base a los números impares.
        # P.e. Si hay 2 dimensiones (len(bins) == 2): np.arange(1, 2 * 2, 2) -> np.arange(1, 4, 2) devuelve [1, 3]
        #      Si la dimensión 1 se desplaza en 1 unidad, en la dimensión 2 se desplazará en 3 unidades.
        # P.e. Si hay 3 dimensiones (len(bins) == 3): np.arange(1, 2 * 3, 2) -> np.arange(1, 6, 2) devuelve [1, 3, 5]
        # P.e. Si hay 4 dimensiones (len(bins) == 4): np.arange(1, 2 * 4, 2) -> np.arange(1, 8, 2) devuelve [1, 3, 5, 7]
        # Y así sucesivamente.
        # displacement_vector se ajusta automáticamente generando un array de números impares
        # Estos valores se usan posteriormente para calcular los desplazamientos específicos en cada dimensión al crear las tilings (rejillas).
        # ¿Por qué esos valores? Porque son los recomendados: los primeros números impares.
        displacement_vector = np.arange(1, 2 * len(bins), 2, dtype=np.float64)


        tilings = []  # Lista que almacenará todas las tilings generadas.
        for i in range(1, n_tilings + 1):
            # Para cada tiling 'i', se calculan nuevos límites 'low_i' y 'high_i' con un desplazamiento aleatorio.
            # El desplazamiento aleatorio se basa en el 20% de los límites originales.
            low_i = low - random.random() * 0.2 * low
            high_i = high + random.random() * 0.2 * high

            # Vamos a calcular el desplazamiento específico para cada dimensión y cada mosaico.

            # Antes calculamos displacement_vector, que nos indica el desplazamiento en cada dimensión.
            # Como tenemos varios mosaicos, cada uno se tendrá que desplazar
            # en la mismas cantidades con respecto al mosaico anterior.
            # Esto se puede conseguir multiplicando el displacement_vector por el número de mosaico (i),
            # pero se toma el módulo n (número total de mosaicos).
            # De esta forma el desplazamiento de cada mosaico es diferente, dentro del rango [0, n-1]

            # P.e. Para n=4 mosaicos, y dos dimensiones, los vectores de desplazamiento de cada mosaico son:
            # i = 1: [1, 3] = [1, 3] * 1 % 4 = [1, 3] % 4
            # i = 2: [2, 2] = [1, 3] * 2 % 4 = [2, 6] % 4
            # i = 3: [3, 1] = [1, 3] * 3 % 4 = [3, 9] % 4
            # i = 4: [0, 0] = [1, 3] * 4 % 4 = [4, 12] % 4
            displacements = (displacement_vector * i) % n_tilings

            # Pero hay que escalar el desplazamiento a unidades reales en cada dimensión.
            # Para ello necesitamos calcular el tamaño de cada segmento (intervalo) en cada dimensión.
            segment_sizes = (high_i - low_i) / bins

            # Entonces usamos una fracción del tamaño del segmento para desplazar cada mosaico.
            # La fracción del tamaño del segmento viene dado por el tamaño del segmento dividido por el número de mosaicos.
            # Por ejemplo, si el tamaño de la celda es 0.5 en la primera dimensión y se consideran n=4 mosaicos, la fracción es 0.5/4=0.125
            # Según se ha calculado anteriormente, en el vector de desplazamiento,
            # la primera dimensión se desplaza en 1, 2, 3 y 0 unidades para los mosaicos 1, 2, 3, y 4, respectivamente.
            # Como la unidad es 0.125, entonces la primera dimensión de cada mosaico se desplaza en las cantidades:
            # 0.125 = 1 * 0.125,  0.25 = 2 * 0.125, 0.375 = 3 * 0.125, y  0 = 0 * 0.125.
            # Lo mismo se haría con el resto de dimensiones. En forma vectorial:
            # Es decir, el desplazamiento de cada mosaico en la primera dimensión es:
            # Tiling 1, [1, 3]: [1 * 0.125, 3 * 0.05] = [0.125, 0.15]
            # Tiling 2, [2, 2]: [2 * 0.125, 2 * 0.05] = [0.25, 0.10]
            # Tiling 3, [3, 1]: [3 * 0.125, 1 * 0.05] = [0.375, 0.05]
            # Tiling 4  [0, 0]: [0 * 0.125, 0 * 0.05] = [0, 0]
            displacements = displacements * (segment_sizes / n_tilings)

            # Se aplican los desplazamientos a los límites inferiores y superiores.
            low_i += displacements
            high_i += displacements

            # Para cada dimensión, se crean los buckets que dividen el intervalo de low_i a high_i en 'bins' partes,
            # generando 'l-1' puntos (límites) para cada dimensión.
            buckets_i = []
            for dim, (j, k, l) in enumerate(zip(low_i, high_i, bins)):
                if passthrough_mask[dim]:
                    buckets_i.append(None)
                else:
                    buckets_i.append(np.linspace(j, k, l - 1))

            # Se añade la tiling actual (lista de buckets para cada dimensión) a la lista de tilings.
            tilings.append(buckets_i)

        return tilings  # Retorna la lista completa de tilings.


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    env.reset(seed=0)

    tilings = 4
    bins = np.array([4, 4, 4, 4, 4, 4, 2, 2])
    low = env.observation_space.low
    high = env.observation_space.high
    tcenv = TileCodingEnv(env, bins=bins, low=low, high=high)
    print("Se muestran los 4 mosaicos")
    print(tcenv.tilings)

    print(f"El espacio de observaciones original es: {env.observation_space}, \n\
    Un estado para este espacio es: {env.step(env.action_space.sample())}")
    print(f"El espacio de estados modificado es: {tcenv.observation_space}, \n\
    Un estado para este nuevo espacio es: {tcenv.step(tcenv.action_space.sample())[0]} \n\
    Cada pareja es la 'celda' correspondiente a cada mosaico")

    #plot first 2 dimensions of the tilings in one plot (vertical and horizontal lines for region borders)
    import matplotlib.pyplot as plt
    c = ['r', 'g', 'b', 'm']
    for i, tiling in enumerate(tcenv.tilings):
        for b in tiling[0]: # vertical lines for dimension 1
            plt.axvline(x=b, color=c[i], linestyle='--', label='Dim 1' if b == tiling[0][0] else "")
        for b in tiling[1]: # horizontal lines for dimension 2
            plt.axhline(y=b, color=c[i], linestyle='--', label='Dim 2' if b == tiling[1][0] else "")
    plt.title("Tilings (buckets) para las primeras 2 dimensiones")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    margin = 10
    plt.xlim(low[0]-margin, high[0]+margin)
    plt.ylim(low[1]-margin, high[1]+margin)
    plt.legend()
    plt.grid()
    plt.show()