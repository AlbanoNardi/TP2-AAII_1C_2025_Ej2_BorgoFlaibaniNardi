from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

    def get_state(self, state):
        """Discretiza el estado continuo en un estado discreto para Q-learning."""

        # Variables básicas del estado
        player_y = state['player_y']
        player_vel = state['player_vel']
        pipe_dist = state['next_pipe_dist_to_player']
        next_pipe_dist = state['next_next_pipe_dist_to_player']
        pipe_top_y = state['next_pipe_top_y']
        pipe_bottom_y = state['next_pipe_bottom_y']
        next_pipe_top_y = state['next_next_pipe_top_y']
        
        # Feature engineering mejorado
        pipe_center_y = (pipe_top_y + pipe_bottom_y) / 2
        pipe_gap_height = pipe_bottom_y - pipe_top_y
        player_pos = int(player_y > pipe_center_y) 
                                
        # Posición relativa del jugador respecto al tubo
        player_relative_y = (player_y - pipe_center_y) / pipe_gap_height
        if player_relative_y > 0.3:
            player_zone = 2  # Arriba del centro
        elif player_relative_y < -0.3:
            player_zone = 0  # Abajo del centro
        else:
            player_zone = 1  # En el centro (zona segura)
        
        # Distancia normalizada al próximo tubo
        pipe_dist_norm = pipe_dist / (next_pipe_dist - pipe_dist)
        if pipe_dist_norm > 0.66:
            dist_zone = 2  # Lejos
        elif pipe_dist_norm > 0.33:
            dist_zone = 1  # Medio
        else:
            dist_zone = 0  # Cerca
        
        # Diferencia de altura entre tubos consecutivos (tendencia del terreno)
        pipe_height_trend = np.sign(next_pipe_top_y - pipe_top_y)

        player_vel_norm = (player_vel + 10) / 20  # Velocidad normalizada

        player_vel_bin = int(np.clip(player_vel_norm * 4, 0, 3))

        
        return (
            player_pos,
            player_vel_bin, 
            int(pipe_height_trend >= 0),
            player_zone,
            dist_zone
        )

    def act(self, state):
        """Elige una acción basada en el estado actual utilizando la red neuronal."""
        # Obtener el estado discretizado
        state_values = self.get_state(state)
        state_values = np.array(state_values, dtype=np.float32).reshape(1, -1)
        actions = self.model(state_values, training = False).numpy()[0]
        action_index = np.argmax(actions)

        return self.actions[action_index]
