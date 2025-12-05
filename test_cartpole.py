import gymnasium as gym
import numpy as np
from dqn_agent import JetsonDQNAgent # Tu archivo con el agente PyTorch

# 1. Crear entorno CartPole
env = gym.make("CartPole-v1", render_mode="human") # 'human' para ver la ventanita (si tienes monitor en la Jetson)

# 2. Configurar dimensiones para tu Agente
N_INPUTS = env.observation_space.shape[0] # Es 4
M_OUTPUTS = env.action_space.n            # Es 2

agent = JetsonDQNAgent(n_input_dim=N_INPUTS, m_output_dim=M_OUTPUTS)
print(f"Entrenando en CartPole. Entradas: {N_INPUTS}, Acciones: {M_OUTPUTS}")

EPISODES = 150

for e in range(EPISODES):
    # Reiniciar entorno
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Tu agente decide
        action = agent.choose_action(state)
        
        # El entorno responde
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Importante: CartPole da +1 por cada frame vivo. 
        # Si se cae, el episodio acaba.
        
        # Guardar y entrenar
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()
        
        state = next_state
        total_reward += reward
        
    print(f"Episodio {e}: Puntos {total_reward} - Epsilon {agent.epsilon:.2f}")

env.close()