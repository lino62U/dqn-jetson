cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dqn_jetson)

# Encuentra LibTorch
# Asegúrate de configurar CMAKE_PREFIX_PATH al ejecutar cmake si no está en path estándar
find_package(Torch REQUIRED)

add_executable(dqn_jetson dqn_jetson.cpp)
target_link_libraries(dqn_jetson "${TORCH_LIBRARIES}")
set_property(TARGET dqn_jetson PROPERTY CXX_STANDARD 14)