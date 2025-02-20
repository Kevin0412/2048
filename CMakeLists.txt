cmake_minimum_required(VERSION 3.22.1)

# Set the project name
project(2048 LANGUAGES CXX)

set(Torch_DIR ~/libtorch/share/cmake/Torch)

# Find the Torch package
find_package(Torch REQUIRED)

# Add the game2048 executable
add_executable(game2048 game2048.cpp)
add_executable(test_model test_model.cpp)

# Link Torch to the project
target_link_libraries(test_model "${TORCH_LIBRARIES}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add definitions for Torch
add_definitions(${TORCH_DEFINITIONS})
