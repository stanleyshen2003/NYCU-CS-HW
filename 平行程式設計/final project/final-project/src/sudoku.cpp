#include "sudoku.h"
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <ctime>
#include <random>

void Sudoku::allocateGrid() {
    grid = new uint8_t*[size];
    for (int i = 0; i < size; ++i) {
        grid[i] = new uint8_t[size]();
    }
}

void Sudoku::deallocateGrid() {
    if (grid) {
        for (int i = 0; i < size; ++i) {
            delete[] grid[i];
        }
        delete[] grid;
        grid = nullptr;
    }
}

Sudoku::Sudoku() : size(0), grid(nullptr) {}

Sudoku::Sudoku(int size) : size(size), grid(nullptr) {
    if (size > 0) allocateGrid();
}

Sudoku::Sudoku(int size, uint8_t** grid) : size(size), grid(nullptr) {
    if (size > 0) {
        allocateGrid();
        for (int i = 0; i < size; ++i) {
            std::copy(grid[i], grid[i] + size, this->grid[i]);
        }
    }
}

Sudoku::Sudoku(const Sudoku& other) : size(other.size), grid(nullptr) {
    if (size > 0) {
        allocateGrid();
        for (int i = 0; i < size; ++i) {
            std::copy(other.grid[i], other.grid[i] + size, grid[i]);
        }
    }
}

Sudoku::Sudoku(Sudoku&& other) noexcept : size(other.size), grid(other.grid) {
    other.size = 0;
    other.grid = nullptr;
}

Sudoku::~Sudoku() {
    deallocateGrid();
}

Sudoku& Sudoku::operator=(const Sudoku& other) {
    if (this != &other) {
        deallocateGrid();
        size = other.size;
        if (size > 0) {
            allocateGrid();
            for (int i = 0; i < size; ++i) {
                std::copy(other.grid[i], other.grid[i] + size, grid[i]);
            }
        }
    }
    return *this;
}

Sudoku& Sudoku::operator=(Sudoku&& other) noexcept {
    if (this != &other) {
        deallocateGrid();
        size = other.size;
        grid = other.grid;
        other.size = 0;
        other.grid = nullptr;
    }
    return *this;
}

void Sudoku::copyFrom(const Sudoku& other) {
    *this = other;
}

void Sudoku::loadSudoku(const std::string& filename) {
    std::ifstream file(filename);
    int gridSize;
    
    if (!file || !(file >> gridSize) || gridSize <= 0) {
        std::cerr << "Invalid file or grid size." << std::endl;
        return;
    }

    if (size != gridSize) {
        deallocateGrid();
        size = gridSize;
        allocateGrid();
    }

    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            int value;
            file >> value;
            grid[row][col] = static_cast<uint8_t>(value);
        }
    }
}

void Sudoku::print() const {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            std::printf("%3d", static_cast<int>(grid[row][col]));
        }
        std::printf("\n");
    }
}

bool Sudoku::isValid() const {
    for (int row = 0; row < size; ++row) {
        std::unordered_set<int> rowSet;
        for (int col = 0; col < size; ++col) {
            int value = grid[row][col];
            if (value < 1 || value > size || !rowSet.insert(value).second) {
                return false;
            }
        }
    }

    for (int col = 0; col < size; ++col) {
        std::unordered_set<int> colSet;
        for (int row = 0; row < size; ++row) {
            int value = grid[row][col];
            if (value < 1 || value > size || !colSet.insert(value).second) {
                return false;
            }
        }
    }

    int subGridSize = static_cast<int>(std::sqrt(size));
    for (int boxRow = 0; boxRow < size; boxRow += subGridSize) {
        for (int boxCol = 0; boxCol < size; boxCol += subGridSize) {
            std::unordered_set<int> boxSet;
            for (int row = 0; row < subGridSize; row++) {
                for (int col = 0; col < subGridSize; col++) {
                    int value = grid[boxRow + row][boxCol + col];
                    if (value < 1 || value > size || !boxSet.insert(value).second) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

void Sudoku::random_empty_cells(int empty_cells) {
    std::vector<int> empty_indices;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            empty_indices.push_back(i * size + j);
        }
    }
    std::srand(std::time(nullptr));
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(empty_indices.begin(), empty_indices.end(), g);
    
    for (int i = 0; i < empty_cells; ++i) {
        int row = empty_indices[i] / size;
        int col = empty_indices[i] % size;
        grid[row][col] = 0;
    }
}