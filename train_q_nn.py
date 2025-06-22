import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
X = np.array(X)
y = np.array(y)

# --- Definir la red neuronal ---
model = keras.Sequential([
    # COMPLETAR: Definir la arquitectura de la red neuronal
    layers.Input(shape=(5,)),
    layers.Dense(15, activation='relu'),
    # salida con 2 neuronas para las acciones (arriba, nada)
    layers.Dense(2, activation = 'linear') # usamos 'linear' porque es una regresión
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- Entrenar la red neuronal ---
model.fit(X, y, epochs = 200, batch_size = 32)
# --- Mostrar resultados del entrenamiento ---
# Completar: Imprimir métricas de entrenamiento


# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas
model.save('flappy_q_nn_model.keras')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
