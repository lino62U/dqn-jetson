# main_jetson.py
import numpy as np
import time
from dqn_jetson import JetsonDQNAgent

# --- CONFIGURACIÓN GENÉRICA ---
INPUT_N = 8    # Tu vector de entrada (sensores, datos, etc.)
OUTPUT_M = 3   # Tus acciones posibles (motores, decisiones, etc.)
MAX_EPISODES = 100

# Instanciar agente
agent = JetsonDQNAgent(n_input_dim=INPUT_N, m_output_dim=OUTPUT_M)

print(f"Iniciando entrenamiento en Jetson. Entrada: {INPUT_N}, Salida: {OUTPUT_M}")

# Bucle de simulación (reemplaza esto con tu bucle de hardware real)
for episode in range(MAX_EPISODES):
    
    # 1. Obtener estado inicial (simulado)
    # En tu caso real: state = leer_sensores_jetson()
    state = np.random.rand(INPUT_N) 
    
    total_reward = 0
    done = False
    step = 0
    
    start_time = time.time()
    
    while not done and step < 200:
        # 2. Agente decide acción
        action_index = agent.choose_action(state)
        
        # 3. Aplicar acción al hardware y obtener respuesta
        # En real: motor.move(action_index); next_state = leer_sensores()
        # Simulamos la respuesta del entorno:
        next_state = np.random.rand(INPUT_N)
        reward = np.random.choice([0.1, -0.1, 1.0]) # Recompensa simulada
        if reward == 1.0: 
            done = True
            
        # 4. Guardar en memoria
        agent.store_transition(state, action_index, reward, next_state, done)
        
        # 5. Entrenar (Optimización: entrenar cada X pasos o al final para no bloquear el bucle de control real)
        loss = agent.learn()
        
        state = next_state
        total_reward += reward
        step += 1
        
        # Actualizar red objetivo ocasionalmente
        if step % 50 == 0:
            agent.update_target_model()

    print(f"Episodio {episode+1}: Recompensa {total_reward:.2f} | Epsilon {agent.epsilon:.2f} | Pasos {step}")

# Guardar modelo final
agent.save("mi_modelo_jetson.h5")