import gymnasium as gym
import numpy as np
import time
from dqn_agent import JetsonDQNAgent 

# Debe coincidir con el nombre usado en train.py
MODEL_FILE = "cartpole_brain.pth"

def main():
    print("\n" + "="*50)
    print("🎬 INICIANDO PRUEBA VISUAL (Inferencia Pura)")
    print("="*50)
    
    # Intentamos abrir modo gráfico ('human'). Si falla (ej. SSH), usa consola.
    try:
        env = gym.make("CartPole-v1", render_mode="human")
        print("Modo gráfico activado.")
    except Exception:
        print("⚠️ No se pudo abrir ventana gráfica. Usando modo consola.")
        env = gym.make("CartPole-v1", render_mode=None)

    N_INPUTS = env.observation_space.shape[0]
    M_OUTPUTS = env.action_space.n

    # Instanciar agente
    agent = JetsonDQNAgent(n_input_dim=N_INPUTS, m_output_dim=M_OUTPUTS)
    
    # CARGAR CEREBRO PRE-ENTRENADO
    agent.load(MODEL_FILE)
    
    # IMPORTANTE: Desactivar aleatoriedad
    agent.epsilon = 0.0
    print("Modo 'Greedy' activado (Epsilon=0). El agente usará solo lo aprendido.")
    
    TEST_EPISODES = 5
    
    for e in range(TEST_EPISODES):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n▶️ Test {e+1}...", end="")
        
        while not done:
            # Elegir acción (sin aprendizaje ni aleatoriedad)
            action = agent.choose_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            # Pequeña pausa para visualización humana (opcional)
            # time.sleep(0.02) 
            
        print(f" Terminado. Puntuación Final: {total_reward}")
        
        # Pausa entre episodios
        time.sleep(1)

    env.close()
    print("\n✅ Pruebas finalizadas.")

if __name__ == "__main__":
    main()