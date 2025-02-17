#include "game.hpp"
#include <torch/script.h> // One-stop header.
#include <iostream>

int main() {
    setRandomSeed();
    char actionspace[] = {'w', 'a', 's', 'd'};
    // Load the model.
    torch::jit::script::Module module;
    module = torch::jit::load("../best_policy_net.pt", torch::kCPU);
    module.eval();

    // Initialize the game.
    Game game;
    game.reset();
    game.newBlock();
    game.newBlock();

    // Play the game using the model.
    while (1) {
        // Get the current state of the game.
        std::vector<double> state = game.board.normalize();

        // Convert state to a 2D tensor.
        torch::Tensor state_tensor = torch::tensor(state, torch::kFloat32).reshape({1, 1, 4, 4});

        // Run the model to get the action.
        torch::Tensor action_tensor = module.forward({state_tensor}).toTensor();
        int action = action_tensor.argmax(1).item<int>();

        // Perform the action in the game.
        if (game.board.isMoveable(actionspace[action])) {
            game.move2(actionspace[action]);
            game.newBlock();
        } else {
            // Select a random action if the chosen action is invalid.
            int random_action = rand() % 4;
            while (!game.board.isMoveable(actionspace[random_action])) {
                random_action = rand() % 4;
            }
            game.move2(actionspace[random_action]);
            game.newBlock();
        }

        // Print the current state of the game.
        std::cout << action << std::endl << game.board.toString();
        if (game.isEnd()) {
            break;
        }
    }

    std::cout << "Game over! Your score: " << game.score << std::endl;
    return 0;
}