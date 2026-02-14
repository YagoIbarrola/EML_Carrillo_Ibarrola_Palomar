# Problema del Bandido de k-Brazos

## Información

- **Alumnos:** Carrillo Ibáñez, Ginés; Ibarrola Lapeña, Yago; Palomar Peña, Aarón
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** 1

---

## Descripción

Este proyecto desarrolla un estudio experimental del problema del bandido de k-brazos en distintos entornos estacionarios.

Se implementan y comparan distintas familias de algoritmos:

- Métodos Greedy: $\epsilon$-greedy, $\epsilon$-decaimiento
- Métodos UCB: UCB1, UCB2
- Métodos de Gradiente: Softmax

Los algoritmos se evalúan sobre distintos tipos de brazos:

- Distribución Bernoulli
- Distribución Binomial
- Distribución Normal

Se analizan las siguientes gráficas y métricas:

- Recompensa promedio
- Porcentaje de selección del brazo óptimo
- Estadísticas por brazo
- Regret acumulado


El objetivo es comparar empíricamente el comportamiento exploración-explotación de cada familia de métodos.

---

## Estructura del repositorio

```
k_brazos/
│
├── data/                     # Datos auxiliares (si procede)
├── docs/                     # Documentación adicional
├── src/
│   ├── algorithms/           # Implementación de algoritmos
│   ├── arms/                 # Implementación de brazos
│   ├── plotting/             # Funciones de visualización
│   ├── bandit_experiment.ipynb 
│   ├── comparation_experiment.ipynb
│   ├── epsilon_greedy.ipynb 
│   ├── requirements.txt
│   ├── softmax_experiment.ipynb
│   ├── ucb.ipynb
│   └── utils.py
|── tests/                    # Tests (si procede)
├── main.ipynb                # Notebook principal
├── README.md
```

---

## Reproducibilidad

* Se utiliza semilla fija (42)
* Todos los notebooks están ejecutados

---

## Tecnologías Utilizadas

* Python 3.10
* NumPy
* Matplotlib
* Jupyter Notebook
* Google Colab

---

## Notas Importantes

* Los estudios experimentales forman parte de la evaluación.
* Solo se incluyen en el informe PDF los resultados más relevantes.
* El repositorio contiene experimentación adicional no incluida en el documento escrito.