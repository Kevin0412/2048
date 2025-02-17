#include "game.hpp"

int main() {
    srand(static_cast<unsigned>(time(0)));
    Game game;
    game.playInTerminal();
    return 0;
}