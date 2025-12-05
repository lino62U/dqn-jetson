#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <memory>
#include <cmath>

// --- CONFIGURACIÓN ---
const int INPUT_N = 8;
const int OUTPUT_M = 3;
const int BATCH_SIZE = 256;
const int MAX_MEMORY = 50000;
const double GAMMA = 0.95;
const double EPSILON_DECAY = 0.995;
const double EPSILON_MIN = 0.01;
const double LEARNING_RATE = 0.00025;

// Verificar si hay GPU disponible (Jetson CUDA)
torch::Device device(torch::kCPU);

void setup_jetson_hardware() {
    if (torch::cuda::is_available()) {
        std::cout << "✅ GPU (CUDA) Detectada. Usando hardware de Jetson." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "⚠️ PRECAUCIÓN: No se detectó GPU. Ejecutando en CPU." << std::endl;
    }
}

// --- 2. MODELO DE RED NEURONAL ---
struct DQNImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, out{nullptr};

    DQNImpl(int input_dim, int output_dim) {
        // Equivalente a layers.Dense
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 512));
        fc3 = register_module("fc3", torch::nn::Linear(512, 256));
        out = register_module("out", torch::nn::Linear(256, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        // Salida lineal
        return out->forward(x);
    }
};
TORCH_MODULE(DQN); // Macro para crear punteros inteligentes automáticamente

// --- 3. BUFFER DE REPETICIÓN ---
struct Transition {
    torch::Tensor state;
    int action;
    float reward;
    torch::Tensor next_state;
    bool done;
};

class ReplayBuffer {
public:
    std::deque<Transition> buffer;
    
    void add(torch::Tensor state, int action, float reward, torch::Tensor next_state, bool done) {
        if (buffer.size() >= MAX_MEMORY) {
            buffer.pop_front();
        }
        // Guardar tensores en CPU para no llenar la VRAM, mover a GPU solo al entrenar
        buffer.push_back({state.cpu(), action, reward, next_state.cpu(), done});
    }

    std::vector<Transition> sample(int batch_size) {
        std::vector<Transition> batch;
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch),
                    batch_size, std::mt19937{std::random_device{}()});
        return batch;
    }

    size_t size() { return buffer.size(); }
};

// --- 4. AGENTE DQN ---
class JetsonDQNAgent {
public:
    DQN model{nullptr};
    DQN target_model{nullptr};
    torch::optim::Adam optimizer{nullptr};
    ReplayBuffer memory;
    double epsilon = 1.0;

    JetsonDQNAgent(int input_dim, int output_dim)
        : model(input_dim, output_dim),
          target_model(input_dim, output_dim),
          optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE)) {
        
        // Mover modelos a la GPU (Jetson)
        model->to(device);
        target_model->to(device);
        
        update_target_model();
    }

    void update_target_model() {
        // Copia profunda de pesos: target = model
        // En C++ esto se hace serializando o copiando parámetros manualmente
        // Forma rápida en LibTorch:
        torch::autograd::GradMode::set_enabled(false);  // No necesitamos gradientes para la copia
        auto target_params = target_model->named_parameters();
        auto model_params = model->named_parameters();
        
        for (auto& val : model_params) {
            auto name = val.key();
            auto* target_param = target_params.find(name);
            if (target_param != nullptr) {
                target_param->copy_(val.value());
            }
        }
        torch::autograd::GradMode::set_enabled(true);
    }

    int choose_action(torch::Tensor state) {
        // Epsilon greedy
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (dis(gen) < epsilon) {
            return std::rand() % OUTPUT_M;
        }

        model->eval(); // Modo inferencia
        torch::NoGradGuard no_grad; // Desactivar cálculo de gradientes
        
        state = state.to(device).unsqueeze(0); // [1, N]
        torch::Tensor q_values = model->forward(state);
        return q_values.argmax(1).item<int>();
    }

    void store_transition(torch::Tensor state, int action, float reward, torch::Tensor next_state, bool done) {
        memory.add(state, action, reward, next_state, done);
    }

    float learn() {
        if (memory.size() < BATCH_SIZE) return 0.0f;

        auto batch = memory.sample(BATCH_SIZE);

        // Preparar batch tensors
        std::vector<torch::Tensor> state_batch, next_state_batch;
        std::vector<int64_t> action_batch;
        std::vector<float> reward_batch, done_batch;

        for (const auto& t : batch) {
            state_batch.push_back(t.state);
            next_state_batch.push_back(t.next_state);
            action_batch.push_back(t.action);
            reward_batch.push_back(t.reward);
            done_batch.push_back(t.done ? 1.0f : 0.0f);
        }

        // Stack y mover a GPU
        auto states = torch::stack(state_batch).to(device); // [Batch, N]
        auto next_states = torch::stack(next_state_batch).to(device);
        auto actions = torch::tensor(action_batch).to(device).to(torch::kInt64).unsqueeze(1); // [Batch, 1]
        auto rewards = torch::tensor(reward_batch).to(device).unsqueeze(1);
        auto dones = torch::tensor(done_batch).to(device).unsqueeze(1);

        model->train(); // Modo entrenamiento

        // 1. Calcular Q actual
        // gather selecciona los valores Q de las acciones que tomamos
        torch::Tensor q_values = model->forward(states).gather(1, actions);

        // 2. Calcular Q target
        torch::Tensor next_q_values = target_model->forward(next_states).max(1).values().unsqueeze(1).detach();
        torch::Tensor expected_q_values = rewards + (GAMMA * next_q_values * (1.0 - dones));

        // 3. Calcular Loss (Huber Loss equivalente a SmoothL1Loss en PyTorch)
        torch::Tensor loss = torch::nn::functional::smooth_l1_loss(q_values, expected_q_values);

        // 4. Backpropagation
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Decaer epsilon
        if (epsilon > EPSILON_MIN) epsilon *= EPSILON_DECAY;

        return loss.item<float>();
    }
    
    void save(std::string path) {
        torch::save(model, path);
        std::cout << "Modelo guardado en " << path << std::endl;
    }
};

// --- MAIN (SIMULACIÓN) ---
int main() {
    setup_jetson_hardware();
    std::srand(std::time(nullptr)); // Semilla random

    JetsonDQNAgent agent(INPUT_N, OUTPUT_M);
    
    std::cout << "Iniciando entrenamiento C++ en Jetson..." << std::endl;

    for (int episode = 0; episode < 100; ++episode) {
        // Estado inicial aleatorio simulado
        torch::Tensor state = torch::rand({INPUT_N});
        
        float total_reward = 0;
        bool done = false;
        int step = 0;

        while (!done && step < 200) {
            int action = agent.choose_action(state);

            // Simular entorno
            torch::Tensor next_state = torch::rand({INPUT_N});
            // Recompensa aleatoria simple para test
            float r_val = (std::rand() % 100) / 100.0f; 
            float reward = (r_val > 0.9) ? 1.0f : (r_val < 0.1 ? -0.1f : 0.1f);
            
            if (reward == 1.0f) done = true;

            agent.store_transition(state, action, reward, next_state, done);
            
            float loss = agent.learn();

            state = next_state;
            total_reward += reward;
            step++;

            if (step % 50 == 0) agent.update_target_model();
        }
        
        std::cout << "Episodio " << episode + 1 
                  << " | Recompensa: " << total_reward 
                  << " | Epsilon: " << agent.epsilon 
                  << std::endl;
    }

    agent.save("modelo_jetson_cpp.pt");
    return 0;
}