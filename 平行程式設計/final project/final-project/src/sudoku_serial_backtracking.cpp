#include <iostream>
#include <cmath>
#include "sudoku_serial_backtracking.h"

// extern Sudoku* result;

void SerialBacktrackingSolver::init(const Sudoku& sudoku) {
    result = new Sudoku();
    result->size = sudoku.size;
    result->grid = new uint8_t*[sudoku.size];
    for (int i = 0; i < sudoku.size; ++i) {
        result->grid[i] = new uint8_t[sudoku.size];
        for (int j = 0; j < sudoku.size; ++j) {
            result->grid[i][j] = sudoku.grid[i][j];
        }
    }
}

void SerialBacktrackingSolver::solve() {
    backtracking();
}
bool SerialBacktrackingSolver::backtracking() {
    int row, col;

    // Find the next empty cell (0 indicates an empty cell)
    if (!find_empty(row, col)) {
        return true;  // No empty cells left, puzzle is solved
    }
    // std::cout <<"(row, col): " << row << ", " << col << "\n";
    // result->print();
    // Try placing numbers 1 to 9 in the empty cell
    for (int num = 1; num <= result->size; ++num) {
        if (is_valid(row, col, num)) {
            result->grid[row][col] = num;
            
            // Recursively attempt to solve the rest of the puzzle
            if (backtracking()) {
                return true;
            }

            // Backtrack if placing num didn't lead to a solution
            result->grid[row][col] = 0;
        }
    }
    return false;  // Trigger recursion
}
bool SerialBacktrackingSolver::is_valid(int row, int col, int num) const {
    // Check row and column for conflicts
    for (int i = 0; i < result->size; ++i) {
        if (result->grid[row][i] == num || result->grid[i][col] == num) {
            return false;
        }
    }
    int grid_length = sqrt(result->size);
    int startRow = (row / grid_length) * grid_length;
    int startCol = (col / grid_length) * grid_length;
    
    for (int i = 0; i < grid_length; ++i) {
        for (int j = 0; j < grid_length; ++j) {
            if (result->grid[startRow + i][startCol + j] == num) {
                return false;
            }
        }
    }
    return true;
}

void SerialBacktrackingSolver::display() const {
    
    for (int i = 0; i < result->size; ++i) {
        for (int j = 0; j < result->size; ++j) {
            std::cout << static_cast<int>(result->grid[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}

bool SerialBacktrackingSolver::find_empty(int &row, int &col) const {
    for (row = 0; row < result->size; ++row) {
        for (col = 0; col < result->size; ++col) {
            if (result->grid[row][col] == 0) {
                return true;  // Found an empty cell
            }
        }
    }
    return false;  // No empty cells found
}