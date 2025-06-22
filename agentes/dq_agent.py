from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=0, epsilon_decay=0.995, min_epsilon=0, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

    def discretize_state(self, state):
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
        """Elige una acción usando epsilon-greedy sobre la Q-table."""
        discrete_state = self.discretize_state(state)
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = self.q_table[discrete_state]
            return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
