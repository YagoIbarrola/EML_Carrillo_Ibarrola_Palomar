# Aprendizaje por Refuerzo en Entornos Complejos

## Información

* **Alumnos:** Carrillo Ibáñez, Ginés; Ibarrola Lapeña, Yago; Palomar Peña, Aarón
* **Asignatura:** Extensiones de Machine Learning
* **Curso:** 2025/2026
* **Grupo:** 1

---

## Descripción

Este proyecto desarrolla un estudio experimental del aprendizaje por refuerzo en entornos complejos, donde un agente interactúa con un entorno tomando decisiones secuenciales con el objetivo de maximizar la recompensa acumulada.

A diferencia del problema del bandido de k-brazos, en estos escenarios las decisiones dependen del estado del entorno, lo que se modela mediante **Procesos de Decisión de Markov (MDP)**. En la mayoría de los problemas reales el modelo del entorno no es conocido, por lo que el agente debe aprender a partir de la experiencia obtenida durante la interacción con el entorno.

Para ello se utilizan entornos de la librería **Gymnasium**, que permite experimentar con diferentes algoritmos de aprendizaje por refuerzo.

Se implementan y comparan distintos tipos de algoritmos:

### Métodos Tabulares

* Monte Carlo

  * On-policy
  * Off-policy
* Diferencias Temporales

  * SARSA y Expected SARSA
  * Q-Learning y Double Q-Learning

### Métodos con Aproximación de Funciones

* SARSA Semi-Gradiente (con tilings y con RNA)
* Deep Q-Learning

Los algoritmos se evalúan en distintos entornos de **Gymnasium**:

* **FrozenLake**: entorno discreto sencillo para analizar algoritmos Monte Carlo.
* **Taxi-v3**: entorno tabular con dinámica más compleja para estudiar métodos de diferencias temporales.
* **LunarLander-v3**: entorno con **espacio de estados continuo**, que requiere el uso de **aproximación de funciones mediante redes neuronales**.

Se analizan las siguientes métricas y representaciones:

* Recompensa media por episodio
* Evolución de la duración de los episodios
* Estabilidad del aprendizaje
* Convergencia de la política
* Evolución del error de entrenamiento en métodos con redes neuronales

El objetivo es **comparar empíricamente el comportamiento y la estabilidad de distintos algoritmos de aprendizaje por refuerzo en entornos discretos y continuos**.

---

## Estructura del repositorio

```
entornos_complejos/                 # Proyecto: Aprendizaje por refuerzo en entornos complejos
├── README.md
├── main.ipynb
│
├── data/                           # Datos generados durante los experimentos
│   ├── gifs_taxi/                  # Visualizaciones de episodios del entorno Taxi
│   │   ├── QLearning/
│   │   ├── SARSA/
│   │   ├── expectedSARSA/
│   │   ├── mcOnPolicy/
│   │   └── mcOffPolicy/
│   │
│   ├── icons_taxi/                 # Iconos usados para las animaciones
│   │
│   └── results/                    # Resultados de experimentos y modelos entrenados
│       ├── QLearning/
│       ├── SARSA/
│       ├── expectedSARSA/
│       ├── montecarlo/
│       ├── SARSAsemi/
│       ├── SARSADeepLunarLander/
│       └── DeepQLearningLunarLander/
│
├── docs/                           # Documentación extra (si procede)
│
├── src/
│   ├── agents/                     # Implementaciones de agentes de RL
│   │   ├── agent.py
│   │   ├── lunarAgentDeepQLearning.py 
│   │   ├── lunarAgentSARSADeep.py
│   │   ├── lunarAgentSARSASemi.py
|   |   ├── lunarLanderTileCoding.py
|   |   ├── taxiAgentDoubleQLearning.py
│   │   ├── taxiAgentExpectedSARSA.py
│   │   ├── taxiAgentMontecarloOffPolicy.py
│   │   ├── taxiAgentMontecarloOnPolicy.py
│   │   ├── taxiAgentQLearning.py
│   │   └── taxiAgentSARSA.py
│   ├── taxi_gif.py                 # Generación de animaciones del entorno Taxi
│   └── utils.py                    # Funciones auxiliares
├── lunar_comparation.ipynb
├── lunar_SARSA_Deep_QNet.ipynb
├── lunar_SARSAsemi.ipynb
├── lunarlander_deepqlearning.ipynb
├── MonteCarloTodasLasVisitas.ipynb
├── taxi_comparation.ipynb
├── taxi_montecarlo.ipynb
├── taxi_QLearning.ipynb
├── taxi_SARSA.ipynb
└── tests/                          # Notebooks de prueba y experimentación
```

---

## Reproducibilidad

* Se utiliza **semilla fija** para los experimentos.
* Todos los notebooks incluidos en el repositorio están **ejecutados**.

---

## Tecnologías Utilizadas

* Python 3.10
* NumPy
* Matplotlib
* Gymnasium
* PyTorch / TensorFlow (según implementación de redes neuronales)
* Jupyter Notebook
* Google Colab

---

## Notas Importantes

* Los estudios experimentales forman parte de la evaluación de la asignatura.
* En el informe final solo se incluyen los resultados más representativos.
* El repositorio contiene experimentos adicionales y análisis exploratorios no incluidos en el documento final.
