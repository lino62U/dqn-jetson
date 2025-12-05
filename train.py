import gymnasium as gym
import numpy as np
import argparse  # Librería para leer parámetros de consola
from dqn_agent import JetsonDQNAgent 

# Nombre del archivo donde se guardará el cerebro
MODEL_FILE = "cartpole_brain.pth"

def main():
    # 1. Configurar recepción de argumentos
    parser = argparse.ArgumentParser(description='Entrenamiento DQN en Jetson')
    parser.add_argument('--episodes', type=int, default=150, help='Número de episodios a entrenar (por defecto: 150)')
    args = parser.parse_args()

    EPISODES = args.episodes

    print("\n" + "="*50)
    print(f"🚀 INICIANDO ENTRENAMIENTO (Gymnasium CartPole-v1)")
    print(f"📋 Meta: {EPISODES} episodios")
    print("="*50)
    
    # Creamos el entorno SIN renderizar (render_mode=None) para máxima velocidad
    env = gym.make("CartPole-v1", render_mode=None)
    
    # Detectar dimensiones automáticamente
    N_INPUTS = env.observation_space.shape[0] # 4 variables
    M_OUTPUTS = env.action_space.n            # 2 acciones (Izq, Der)

    print(f"Entorno configurado: {N_INPUTS} entradas, {M_OUTPUTS} salidas.")

    # Instanciar agente
    agent = JetsonDQNAgent(n_input_dim=N_INPUTS, m_output_dim=M_OUTPUTS)
    
    for e in range(EPISODES):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. El agente decide
            action = agent.choose_action(state)
            
            # 2. El entorno responde
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 3. Guardamos la experiencia
            agent.store_transition(state, action, reward, next_state, done)
            
            # 4. El agente aprende
            agent.learn()
            
            state = next_state
            total_reward += reward

        # Actualizar la red objetivo periódicamente
        if e % 10 == 0:
            agent.update_target_model()

        # Formato de impresión alineado
        print(f"Episodio {e+1:03d}/{EPISODES} | Puntos: {total_reward:.0f} | Epsilon: {agent.epsilon:.2f}")

    # Al finalizar, guardar el modelo
    print("\n" + "-"*30)
    agent.save(MODEL_FILE)
    env.close()
    print("✅ Entrenamiento finalizado exitosamente.")

if __name__ == "__main__":
    main()