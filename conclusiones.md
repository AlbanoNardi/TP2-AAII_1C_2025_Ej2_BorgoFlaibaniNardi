# Agentes Inteligentes con Q-Learning y Deep Q-Learning en Flappy Bird

## Objetivo

El objetivo de este proyecto es entrenar agentes para resolver videojuegos sencillos utilizando técnicas de aprendizaje por refuerzo. En particular, se implementan y entrenan dos tipos de agentes para el juego **Flappy Bird**, utilizando la librería `PLE (PyGame Learning Environment)`:

- **Ejercicio A:** Agente basado en **Q-Learning** (Q-Table)
- **Ejercicio B:** Agente basado en **Deep Q-Learning** (Red Neuronal)

---

## Ejercicio A: Agente con Q-Learning

### ✅ Implementación del Agente

El agente utiliza una Q-table para aprender las mejores acciones a partir de estados discretizados del entorno. Para ello, se realizó un proceso de **ingeniería de características** que permite transformar el estado continuo del juego en un espacio de estados discretos manejable con un total de 144 posibles estados discretos, resultantes de combinar las distintas variables categorizadas del entorno.

La discretización incluye:
- **Posición relativa del jugador respecto al centro del tubo** (`player_zone`):
  - Abajo del centro
  - Zona segura (centro)
  - Arriba del centro
- **Distancia normalizada al tubo más próximo** (`dist_zone`):
  - Cerca
  - Medio
  - Lejos
- **Tendencia del terreno**, basada en la diferencia de altura entre tubos consecutivos
- **Velocidad vertical del jugador** en 4 bins normalizados
- **Posición binaria del jugador respecto al centro del tubo**

Esta discretización permite representar el estado con una tupla de 5 elementos discretos, haciendo posible indexar la Q-table de manera eficiente.

### 🏋️ Entrenamiento del Agente

Además de las recompensas por pasar tubos (definidas por el entorno), se implementó una **recompensa positiva por mantenerse vivo**:

```python
current_episode_reward += 0.001  # Recompensa por sobrevivir un paso más
````

Esto contribuyó a mejorar la estabilidad del entrenamiento, favoreciendo políticas que maximizan la duración de vida del agente.

### 🧪 Prueba del Agente Entrenado

A continuación se muestran los resultados de cinco episodios de prueba luego del entrenamiento:

```
Episodio 1:  Recompensa = 86.0
Episodio 2:  Recompensa = 219.0
Episodio 3:  Recompensa = 437.0
Episodio 4:  Recompensa = 144.0
Episodio 5:  Recompensa = 13.0
```

Se observa un desempeño variable pero con episodios de alto rendimiento, lo que indica que el agente logró aprender una política efectiva en muchos casos.

---

## Ejercicio B: Agente con Deep Q-Learning

### 🧠 Entrenamiento de la Red Neuronal

Para este agente, se utilizó una red neuronal entrenada para aproximar la Q-table aprendida previamente por el agente basado en Q-learning. La red fue entrenada usando como dataset las tuplas `(estado_discretizado, acción óptima)` extraídas de la Q-table entrenada, lo que permite al modelo generalizar el comportamiento aprendido sin necesidad de explorar nuevamente el entorno.

La red se guardó como un modelo `TensorFlow SavedModel` en el archivo `flappy_q_nn_model.keras`.

### 🤖 Implementación del Agente Neuronal

El agente neuronal, implementado en la clase `NNAgent`, utiliza la misma función de discretización que el agente de Q-learning, manteniendo así la coherencia en la representación del estado. Sin embargo, en lugar de consultar una tabla, el agente pasa el estado discretizado como input a la red neuronal y selecciona la acción con mayor valor Q predicho.

Esto le permite tomar decisiones más rápidas y adaptarse mejor a estados no vistos exactamente durante el entrenamiento.

### 🧪 Prueba del Agente Neuronal

Resultados de los episodios de prueba del agente con red neuronal:
```
Episodio 1: Recompensa = 411.0
Episodio 2: Recompensa = 188.0
Episodio 3: Recompensa = 33.0
Episodio 4: Recompensa = 1097.0
```

Se observan recompensas significativamente altas en algunos episodios, lo que indica que el modelo neuronal logró generalizar correctamente la política aprendida y, en ocasiones, superó el rendimiento del agente basado en tabla.


---

## Ejecución

Para ejecutar y testear un agente, se utiliza el script `test_agent.py`, pasando el tipo de agente mediante el parámetro `--agent`:

```bash
python test_agent.py --agent q_learning
python test_agent.py --agent dqn
```

## Estructura del Proyecto

```
├── .vscode/
│   └── settings.json
├── agentes/
│   ├── __init__.py
│   ├── base.py
│   ├── dq_agent.py
│   ├── manual_agent.py
│   ├── nn_agent.py
│   └── random_agent.py
├── ple/
│   ├── games/
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── doomwrapper.py
│   │   │   └── pygamewrapper.py
│   │   ├── doom/
│   │   │   ├── __init__.py
│   │   │   └── doom.py
│   │   ├── flappybird/
│   │   │   ├── assets/
│   │   │   └── __init__.py
│   │   ├── monsterkong/
│   │   │   ├── LICENSE
│   │   │   ├── __init__.py
│   │   │   ├── board.py
│   │   │   ├── coin.py
│   │   │   ├── fireball.py
│   │   │   ├── ladder.py
│   │   │   ├── monsterPerson.py
│   │   │   ├── onBoard.py
│   │   │   ├── person.py
│   │   │   ├── player.py
│   │   │   └── wall.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── vec2d.py
│   │   ├── __init__.py
│   │   ├── catcher.py
│   │   ├── pixelcopter.py
│   │   ├── pong.py
│   │   ├── primitives.py
│   │   ├── puckworld.py
│   │   ├── raycast.py
│   │   ├── raycastmaze.py
│   │   ├── snake.py
│   │   └── waterworld.py
│   ├── __init__.py
│   └── ple.py
├── test/
│   ├── FlappyScores.png
│   └── NNScores.png
├── .gitignore
├── README.md
├── conclusiones.md
├── flappy_birds_q_table.pkl
├── flappy_q_nn_model.keras
├── requirements.txt
├── test_agent.py
├── train_q_agent.py
└── train_q_nn.py
```

---

## Requisitos

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

---

## Conclusiones

### Ingeniería de Características

Para ambos agentes se aplicó la misma ingeniería de características, permitiendo representar el entorno continuo del juego mediante una tupla de 5 valores discretos:

- Posición vertical relativa del jugador respecto al centro del tubo (3 zonas)
- Velocidad vertical discretizada en 4 niveles
- Tendencia de altura entre tubos consecutivos (ascendente o no)
- Posición binaria del jugador (arriba o abajo del centro)
- Distancia normalizada al siguiente tubo (3 zonas)

Esto da lugar a **144 combinaciones posibles de estado**, lo cual permite construir una Q-table de tamaño razonable y suficientemente expresiva para el agente basado en Q-learning, y también sirve como entrada de bajo dimensionalidad para la red neuronal.

### Comparación de Resultados

Ambos agentes mostraron un rendimiento competente. El agente de Q-learning aprendió una política sólida con recompensas elevadas en muchos episodios. Sin embargo, el agente con red neuronal logró resultados incluso superiores en algunos casos, alcanzando una recompensa de **1097.0** en un episodio, gracias a su capacidad de generalizar más allá de los estados vistos en la tabla.

La inclusión de una pequeña recompensa por supervivencia resultó clave para estabilizar el aprendizaje en ambos modelos.

En conclusión, aunque el Q-learning es eficaz para este entorno simple, el uso de redes neuronales permite mayor flexibilidad y escalabilidad para escenarios más complejos.
