import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# --- 1. CONFIGURACIÓN DE HARDWARE (JETSON) ---
# Detectamos si hay GPU (CUDA) disponible. En la Jetson debería ser 'cuda'.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
    print("   Activando optimizaciones para Jetson (AMP + CuDNN Benchmark)...")
    # Activa el buscador de algoritmos eficientes para el hardware específico (Xavier)
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️  GPU no detectada. Se usará CPU (Lento).")

# --- 2. RED NEURONAL (Puro PyTorch) ---
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) 
            # Sin activación final porque DQN estima valores Q directos (regresión)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. BUFFER DE MEMORIA ---
class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- 4. CLASE AGENTE DQN ---
class JetsonDQNAgent:
    def __init__(self, n_input_dim, m_output_dim):
        self.state_dim = n_input_dim
        self.action_dim = m_output_dim
        
        # Hiperparámetros
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 256 # Ajustar según memoria de la Jetson (128 o 256 va bien)
        self.learning_rate = 0.00025

        self.buffer = ReplayBuffer()

        # Instanciar modelos y moverlos a la GPU (.to(device))
        self.model = DQNNetwork(self.state_dim, self.action_dim).to(device)
        self.target_model = DQNNetwork(self.state_dim, self.action_dim).to(device)
        
        # Sincronizar pesos iniciales
        self.update_target_model()
        self.target_model.eval() # Modo evaluación (no entrena)

        # Optimizador Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Función de pérdida (Huber Loss es muy estable para DQN)
        self.criterion = nn.SmoothL1Loss()

        # [OPTIMIZACIÓN JETSON] GradScaler para Mixed Precision (FP16)
        # Esto permite usar float16 en los Tensor Cores sin perder precisión en gradientes
        self.scaler = torch.cuda.amp.GradScaler()

    def update_target_model(self):
        """Copia los pesos de la red principal a la red objetivo"""
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state):
        """Recibe estado (numpy) -> Devuelve acción (int)"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Inferencia optimizada sin cálculo de gradientes
        with torch.no_grad():
            # Convertir numpy a tensor y enviar a GPU
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_t)
            # Retornar el índice del valor más alto
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        # 1. Obtener lote de datos (en CPU)
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 2. Convertir a Tensores y subir a la GPU (Operación costosa, se hace en bloque)
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

        # 3. Paso de entrenamiento con Mixed Precision (AMP)
        self.optimizer.zero_grad()

        # autocast hace que la red corra en FP16 donde sea posible (rápido en Jetson)
        with torch.cuda.amp.autocast():
            # Predicción actual: Q(s, a)
            q_values = self.model(states_t).gather(1, actions_t)

            # Predicción futura: r + gamma * max Q(s', a')
            with torch.no_grad():
                next_q_values = self.target_model(next_states_t).max(1)[0].unsqueeze(1)
                expected_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))

            # Calcular error
            loss = self.criterion(q_values, expected_q_values)

        # 4. Backpropagation escalado (necesario por FP16)
        self.scaler.scale(loss).backward()
        
        # Clip de gradientes para estabilidad
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filename='dqn_jetson_model.pth'):
        torch.save(self.model.state_dict(), filename)
        print(f"Modelo guardado: {filename}")

    def load(self, filename='dqn_jetson_model.pth'):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            self.update_target_model()
            print(f"Modelo cargado: {filename}")
        else:
            print("No se encontró archivo de modelo para cargar.")