import gymnasium as gym
import numpy as np
import time
from dqn_agent import JetsonDQNAgent 

MODEL_NAME = "cartpole_jetson.pth"

# ==========================================
# 1. ESCENARIO DE ENTRENAMIENTO
# ==========================================
def run_training():
    print("\n" + "="*50)
    print("🚀 FASE 1: ENTRENAMIENTO (Gymnasium CartPole-v1)")
    print("="*50)
    
    # Creamos el entorno SIN renderizar para que entrene rápido
    env = gym.make("CartPole-v1", render_mode=None)
    
    N_INPUTS = env.observation_space.shape[0] # 4
    M_OUTPUTS = env.action_space.n            # 2

    agent = JetsonDQNAgent(n_input_dim=N_INPUTS, m_output_dim=M_OUTPUTS)
    
    EPISODES = 150 # CartPole se resuelve rápido, 150 suele bastar

    for e in range(EPISODES):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. Elegir Acción
            action = agent.choose_action(state)
            
            # 2. Paso en el entorno
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 3. Guardar y Aprender
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward

        # Actualizar red objetivo cada 10 episodios
        if e % 10 == 0:
            agent.update_target_model()

        print(f"Episodio {e+1:03d}/{EPISODES} | Puntos: {total_reward:.0f} | Epsilon: {agent.epsilon:.2f}")

    # Guardar resultado
    agent.save(MODEL_NAME)
    env.close()
    print("✅ Entrenamiento finalizado y modelo guardado.")

# ==========================================
# 2. ESCENARIO DE PRUEBA (VALIDACIÓN)
# ==========================================
def run_testing():
    print("\n" + "="*50)
    print("🎬 FASE 2: PRUEBA VISUAL (Usando el cerebro entrenado)")
    print("="*50)
    
    # Aquí activamos 'human' para VER la ventanita
    # NOTA: Si estás en Jetson sin monitor (SSH), cambiar a render_mode=None
    try:
        env = gym.make("CartPole-v1", render_mode="human")
    except Exception as e:
        print("⚠️ No se pudo abrir ventana gráfica (¿Estás por SSH?). Usando modo consola.")
        env = gym.make("CartPole-v1", render_mode=None)

    N_INPUTS = env.observation_space.shape[0]
    M_OUTPUTS = env.action_space.n

    # Cargar agente
    agent = JetsonDQNAgent(n_input_dim=N_INPUTS, m_output_dim=M_OUTPUTS)
    agent.load(MODEL_NAME)
    
    # MODO EXPERTO: Apagar aleatoriedad y aprendizaje
    agent.epsilon = 0.0
    
    TEST_EPISODES = 5
    
    for e in range(TEST_EPISODES):
        state, info = env.reset()
        done = False
        total_reward = 0
        print(f"▶️ Iniciando Test {e+1}...", end="")
        
        while not done:
            # Solo inferencia, nada de random
            action = agent.choose_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            # Pequeña pausa para que el ojo humano pueda verlo bien si va muy rápido
            # time.sleep(0.01) 
            
        print(f" Terminado. Puntuación: {total_reward}")
        time.sleep(1) # Pausa entre episodios

    env.close()

if __name__ == "__main__":
    run_training()
    
    print("\nCargando fase de prueba en 3 segundos...")
    time.sleep(3)
    
    run_testing()