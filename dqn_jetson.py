import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import os
import time

# --- 1. OPTIMIZACIONES DE HARDWARE (JETSON AGX XAVIER) ---
def setup_jetson_hardware():
    """Configura la GPU para evitar OOM y activar Tensor Cores."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Importante en Jetson: La memoria es compartida. 
                # Esto evita que TF reserve toda la RAM al inicio.
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # ACTIVAR MIXED PRECISION (FP16)
            # Esto utiliza los Tensor Cores de la arquitectura Volta/Xavier
            # Acelera el entrenamiento x2-x3 con mínima pérdida de precisión.
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"✅ GPU Detectada: {len(gpus)}. Mixed Precision (FP16) ACTIVADO.")
            print("   (Optimizaciones para Jetson AGX Xavier aplicadas)")
        except RuntimeError as e:
            print(f"❌ Error configurando GPU: {e}")
    else:
        print("⚠️ PRECAUCIÓN: No se detectó GPU (CUDA). Se ejecutará en CPU (Lento).")

setup_jetson_hardware()

# --- 2. MODELO DE RED NEURONAL ---
def build_model(input_dim, output_dim):
    """
    Crea el modelo DQN.
    input_dim (n): Tamaño del vector de estado.
    output_dim (m): Cantidad de acciones posibles.
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        
        # Capas densas estándar
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        
        # Capa de salida
        # Nota: En Mixed Precision, la salida final debe ser float32 para estabilidad numérica
        layers.Dense(output_dim, activation='linear', dtype='float32') 
    ])
    
    # Optimizador
    # Usamos Adam. El LossScaleOptimizer es manejado automáticamente por la Policy 'mixed_float16' en Keras moderno,
    # pero definimos los hiperparámetros aquí.
    opt = keras.optimizers.Adam(learning_rate=0.00025, clipvalue=1.0)
    
    # Huber Loss es robusto para RL
    model.compile(optimizer=opt, loss='huber')
    return model

# --- 3. BUFFER DE REPETICIÓN ---
class ReplayBuffer:
    def __init__(self, max_size=50000): # Aumentado ligeramente si la RAM de Xavier (32GB) lo permite
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # Asegurar tipos de datos eficientes para ahorrar memoria
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- 4. AGENTE DQN OPTIMIZADO ---
class JetsonDQNAgent:
    def __init__(self, n_input_dim, m_output_dim):
        self.state_dim = n_input_dim  # Entrada N
        self.action_dim = m_output_dim # Salida M
        
        # Hiperparámetros
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 256 # Tamaño de lote bueno para la GPU de Xavier
        
        self.buffer = ReplayBuffer()
        
        # Construcción de modelos
        self.model = build_model(self.state_dim, self.action_dim)
        self.target_model = build_model(self.state_dim, self.action_dim)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        """
        Recibe estado (numpy array shape (N,))
        Retorna índice de acción (int entre 0 y M-1)
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Preprocesamiento rápido para inferencia
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float16) # Usar float16 en entrada
        q_values = self.model(state_tensor, training=False) # training=False es más rápido
        return int(tf.argmax(q_values[0]))

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    # Decorador tf.function con jit_compile=True (XLA) para máxima velocidad en Jetson
    @tf.function(jit_compile=True)
    def _train_step_optimized(self, states, actions, rewards, next_states, dones):
        """Paso de entrenamiento compilado con XLA."""
        
        # Calculamos Q-Targets usando la red objetivo (Target Network)
        q_next = self.target_model(next_states, training=False)
        q_next_max = tf.reduce_max(q_next, axis=1)
        target_q_values = rewards + (self.gamma * q_next_max * (1.0 - dones))

        with tf.GradientTape() as tape:
            # Predicciones de la red principal
            q_values = self.model(states, training=True)
            
            # Seleccionar los Q-values de las acciones tomadas
            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            chosen_q = tf.gather_nd(q_values, indices)
            
            # Calcular pérdida
            loss = tf.keras.losses.Huber()(target_q_values, chosen_q)
            
            # Scaling automático de pérdida si se usa Mixed Precision
            # (Si usamos model.optimizer directamente, Keras maneja el scaling, 
            # pero en GradientTape manual a veces es necesario. 
            # Aquí confiamos en el wrapper automático de Keras Mixed Policy)
            if isinstance(self.model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.model.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss

        # Calcular gradientes
        if isinstance(self.model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.model.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)

        # Aplicar gradientes
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        # Muestreo del buffer
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Conversión a tensores (Eficiente)
        states_t = tf.convert_to_tensor(states, dtype=tf.float16)      # FP16 entrada
        next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float16) # FP16 entrada
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)    # Rewards mejor en FP32
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Ejecutar paso de entrenamiento optimizado
        loss = self._train_step_optimized(states_t, actions_t, rewards_t, next_states_t, dones_t)
        
        # Decrementar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(loss)

    def save(self, name='jetson_model.h5'):
        self.model.save_weights(name)

    def load(self, name='jetson_model.h5'):
        if os.path.exists(name):
            self.model.load_weights(name)
            self.update_target_model()
            print(f"Modelo cargado: {name}")