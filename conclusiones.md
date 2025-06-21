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

## Ejercicio B: Agente con Deep Q-Learning (⚠️ en desarrollo)

### 🔄 Entrenamiento de la Red Neuronal

*Pendiente de implementación.*

### 🤖 Implementación del Agente Neuronal

*Pendiente de implementación.*

### 🧪 Prueba del Agente Neuronal

*Pendiente de implementación.*

---

## Ejecución

Para ejecutar y testear un agente, se utiliza el script `test_agent.py`, pasando el tipo de agente mediante el parámetro `--agent`:

```bash
python test_agent.py --agent q_learning
python test_agent.py --agent dqn
```

## Estructura del Proyecto

```
├── agents/
│   ├── q_learning_agent.py
│   └── dqn_agent.py
├── test_agent.py
├── train_agent.py
├── conclusiones.md
├── README.md
└── requirements.txt
```

---

## Requisitos

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

---

## Créditos
