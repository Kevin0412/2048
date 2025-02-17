#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdlib>

void setRandomSeed(unsigned int seed=std::time(nullptr)) {
    std::srand(static_cast<unsigned int>(seed));
}
class Board {
public:
    Board() : board(4, std::vector<int>(4, 0)) {}

    std::string toString() const {
        std::ostringstream output;
        for (const auto& row : board) {
            for (const auto& cell : row) {
                output << "|";
                if (cell == 0) {
                    output << "    ";
                } else {
                    output << std::setw(4) << (1 << cell);
                }
            }
            output << "|\n";
        }
        return output.str();
    }

    std::vector<int>& operator[](int index) {
        return board[index];
    }

    const std::vector<int>& operator[](int index) const {
        return board[index];
    }

    void rotate(int times) {
        for (int t = 0; t < times; ++t) {
            Board newBoard;
            for (int x = 0; x < 4; ++x) {
                for (int y = 0; y < 4; ++y) {
                    newBoard[x][y] = board[y][3 - x];
                }
            }
            board = newBoard.board;
        }
    }

    bool isEnd() const {
        for (const auto& row : board) {
            for (const auto& cell : row) {
                if (cell == 0) return false;
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (board[i][j] == board[i][j + 1] || board[j][i] == board[j + 1][i]) {
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<int> flatten() const {
        std::vector<int> flat;
        for (const auto& row : board) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return flat;
    }

    int moveScore(char direction) {
        int newScore = 0;
        if (direction == 'd') rotate(2);
        else if (direction == 'w') rotate(1);
        else if (direction == 's') rotate(3);

        for (auto& row : board) {
            int d = 0;
            for (auto& cell : row) {
                if (cell != 0) {
                    if (d == cell) {
                        newScore += 2 * (1 << cell);
                        d = 0;
                    } else {
                        d = cell;
                    }
                }
            }
        }

        if (direction == 'd') rotate(2);
        else if (direction == 'w') rotate(3);
        else if (direction == 's') rotate(1);

        return newScore;
    }

    bool isMoveable(char direction) {
        bool output = false;
        if (direction == 'd') rotate(2);
        else if (direction == 'w') rotate(1);
        else if (direction == 's') rotate(3);

        for (const auto& row : board) {
            int d = -1;
            for (const auto& cell : row) {
                if (cell != 0) {
                    if (cell == d || d == 0) {
                        output = true;
                        break;
                    }
                }
                d = cell;
            }
            if (output) break;
        }

        if (direction == 'd') rotate(2);
        else if (direction == 'w') rotate(3);
        else if (direction == 's') rotate(1);

        return output;
    }

    Board move(char direction) {
        Board newBoard;
        if (direction == 'd') rotate(2);
        else if (direction == 'w') rotate(1);
        else if (direction == 's') rotate(3);

        int e = 0;
        for (auto& row : board) {
            std::vector<int> c;
            for (auto& cell : row) {
                if (cell != 0) c.push_back(cell);
            }
            std::vector<int> d;
            int a = 0;
            while (true) {
                if (c.empty()) break;
                if (a + 1 == c.size()) {
                    d.push_back(c[a]);
                    break;
                }
                if (c[a] != c[a + 1]) {
                    d.push_back(c[a]);
                    a++;
                } else {
                    d.push_back(c[a] + 1);
                    a += 2;
                }
                if (a == c.size()) break;
            }
            while (d.size() < 4) d.push_back(0);
            newBoard[e] = d;
            e++;
        }

        if (direction == 'd') rotate(2);
        else if (direction == 'w') rotate(3);
        else if (direction == 's') rotate(1);

        return newBoard;
    }

    bool maxInCorner() const {
        auto flat = flatten();
        std::sort(flat.rbegin(), flat.rend());
        return flat[0] == board[0][0] || flat[0] == board[0][3] || flat[0] == board[3][0] || flat[0] == board[3][3];
    }

    int maxTile() const {
        auto flat = flatten();
        return 1 << *std::max_element(flat.begin(), flat.end());
    }

    int numBlocks() const {
        auto flat = flatten();
        return std::count_if(flat.begin(), flat.end(), [](int x) { return x != 0; });
    }

    std::vector<double> normalize() const {
        auto flat = flatten();
        std::vector<double> normalized;
        for (auto& cell : flat) {
            normalized.push_back(static_cast<double>(cell) / 17);
        }
        return normalized;
    }

    std::vector<std::vector<double>> normalize2D() const {
        std::vector<std::vector<double>> normalized(4, std::vector<double>(4));
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                normalized[i][j] = static_cast<double>(board[i][j]) / 17;
            }
        }
        return normalized;
    }

    int snake() const {
        int reward = 1;
        auto flat = flatten();
        std::sort(flat.rbegin(), flat.rend());
        int x = 0, y = 0;
        for (int i = 0; i < 4; ++i) {
            bool flag = false;
            for (int j = 0; j < 4; ++j) {
                if (board[i][j] == flat[0]) {
                    x = i;
                    y = j;
                    flag = true;
                    break;
                }
            }
            if (flag) break;
        }
        for (int i = 1; i < 16; ++i) {
            if (flat[i] == 0) break;
            for (int j = -1; j <= 1; j += 2) {
                if (x + j >= 0 && x + j < 4 && board[x + j][y] == flat[i]) {
                    reward++;
                    x += j;
                    break;
                }
                if (y + j >= 0 && y + j < 4 && board[x][y + j] == flat[i]) {
                    reward++;
                    y += j;
                    break;
                }
            }
        }
        return reward;
    }

    int empty_cells() const {
        int count = 0;
        for (const auto& row : board) {
            count += std::count(row.begin(), row.end(), 0);
        }
        return count;
    }

    double reward() const {
        double reward1= maxInCorner() ? (snake() == 1 ? 0 : static_cast<double>(snake() - 1) / (numBlocks() - 1)) : 0;
        double reward2= 1-pow(((14-empty_cells())/14),2);
        return reward1 * reward2;
    }

    std::vector<std::vector<int>> board;
};

class Game {
public:
    Game() : board(), score(0), movements(-2), blocks(16, 0), reward(0) {}

    std::string toString() const {
        std::ostringstream output;
        output << "score:\t" << score << "\n";
        output << "movements:\t" << movements << "\n";
        output << "reward:\t" << reward << "\n";
        output << board.toString();
        return output.str();
    }

    bool isEnd() {
        return board.isEnd();
    }

    void move() {
        int e = 0;
        for (auto& row : board.board) {
            std::vector<int> c;
            for (auto& cell : row) {
                if (cell != 0) c.push_back(cell);
            }
            std::vector<int> d;
            int a = 0;
            while (true) {
                if (c.empty()) break;
                if (a + 1 == c.size()) {
                    d.push_back(c[a]);
                    break;
                }
                if (c[a] != c[a + 1]) {
                    d.push_back(c[a]);
                    a++;
                } else {
                    d.push_back(c[a] + 1);
                    score += 2 * (1 << c[a]);
                    a += 2;
                }
                if (a == c.size()) break;
            }
            while (d.size() < 4) d.push_back(0);
            board[e] = d;
            e++;
        }
    }

    void move2(char direction) {
        if (direction == 'a') {
            move();
        } else if (direction == 'd') {
            board.rotate(2);
            move();
            board.rotate(2);
        } else if (direction == 'w') {
            board.rotate(1);
            move();
            board.rotate(3);
        } else if (direction == 's') {
            board.rotate(3);
            move();
            board.rotate(1);
        }
    }

    void newBlock() {
        int a = 1;
        if (rand() % 10 == rand() % 10) a = 2;
        while (true) {
            int b = rand() % 4;
            int c = rand() % 4;
            if (board[b][c] == 0) {
                board[b][c] = a;
                movements++;
                break;
            }
        }
    }

    void playInTerminal() {
        reset();
        newBlock();
        newBlock();
        while (true) {
            std::cout << "score:\t" << score << "\n";
            std::cout << "movements:\t" << movements << "\n";
            std::cout << "snake:\t" << board.snake() << "\n";
            std::cout << board.toString();
            char direction;
            std::cin >> direction;
            if (board.isMoveable(direction)) {
                move2(direction);
                newBlock();
            } else {
                std::cout << "invalid move\a\n";
            }
            reward += board.reward();
            std::cout << "reward:\t" << reward << "\n";
            if (isEnd()) {
                std::cout << toString();
                break;
            }
        }
        std::cout << "game over\n";
    }

    void reset() {
        board = Board();
        score = 0;
        movements = -2;
        blocks = std::vector<int>(16, 0);
        reward = 0;
    }

    Board board;
    int score;
    int movements;
    std::vector<int> blocks;
    double reward;
};