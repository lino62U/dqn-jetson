import numpy as np
import time
from dqn_agent import JetsonDQNAgent

# --- TUS PARÁMETROS ---
INPUT_N = 8     # Tamaño de tu vector de entrada
OUTPUT_M = 3    # Número de acciones posibles
EPISODES = 500

# Inicializar agente
agent = JetsonDQNAgent(n_input_dim=INPUT_N, m_output_dim=OUTPUT_M)

print(f"Iniciando DQN en Jetson (PyTorch). N={INPUT_N}, M={OUTPUT_M}")

# Bucle principal
for e in range(EPISODES):
    # 1. Leer estado inicial de tus sensores (Simulado aquí)
    # state = leer_sensores_reales() 
    state = np.random.rand(INPUT_N)
    
    total_reward = 0
    done = False
    
    while not done:
        # 2. Agente decide acción
        action = agent.choose_action(state)
        
        # 3. Ejecutar acción en el robot y leer nuevo estado
        # ejecutar_motores(action)
        # next_state = leer_sensores_reales()
        # Simulación:
        next_state = np.random.rand(INPUT_N)
        reward = np.random.choice([-0.1, 0.1, 1]) # Recompensa simulada
        
        if reward == 1: 
            done = True # Condición de término
            
        # 4. Guardar experiencia
        agent.store_transition(state, action, reward, next_state, done)
        
        # 5. Entrenar (Optimización GPU)
        agent.learn()
        
        state = next_state
        total_reward += reward

    # Actualizar red objetivo cada episodio o cada X pasos
    agent.update_target_model()
    
    print(f"Episodio {e+1}/{EPISODES} - Recompensa: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
    
    # Guardar cada 50 episodios
    if (e + 1) % 50 == 0:
        agent.save()