#include <iostream>
#include <cmath>
#include <stack>
#include <omp.h>
#include "sudoku_parallel_backtracking_multiblocks.h"

// extern Sudoku* result;

void ParallelBacktrackingMultiBlocksSolver::init(const Sudoku& sudoku) {
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

void ParallelBacktrackingMultiBlocksSolver::solve() {
    backtracking();
}
void ParallelBacktrackingMultiBlocksSolver::backtracking() {
    std::stack<State> stack; // Stack to store Sudoku states
    int row, col;

    // Find the first empty cell
    if (!find_empty(row, col)) {
        return; // No empty cells, puzzle already solved
    }

    stack.push({row, col, 1}); // Start with the first empty cell and number 1

    while (!stack.empty()) {
        State current = stack.top();
        stack.pop();

        // If the current number is valid, place it
        if (current.num <= result->size && is_valid(current.row, current.col, current.num)) {
            result->grid[current.row][current.col] = current.num;

            // Find the next empty cell
            if (!find_empty(row, col)) {
                return; // Puzzle solved
            }

            // Push the current state back with the next number for backtracking
            stack.push({current.row, current.col, current.num + 1});

            // Push the next empty cell with number 1
            stack.push({row, col, 1});
        } else if (current.num <= result->size) {
            // If not valid, try the next number for the current cell
            stack.push({current.row, current.col, current.num + 1});
        } else {
            // Backtrack if all numbers have been tried
            result->grid[current.row][current.col] = 0;
        }
    }
}

bool ParallelBacktrackingMultiBlocksSolver::is_valid(int row, int col, int num) const {
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

bool ParallelBacktrackingMultiBlocksSolver::find_empty(int &row, int &col) const {
    for (row = 0; row < result->size; ++row) {
        for (col = 0; col < result->size; ++col) {
            if (result->grid[row][col] == 0) {
                return true;
            }
        }
    }
    return false;
}

void ParallelBacktrackingMultiBlocksSolver::copy_result(const Sudoku* local_result) {
    for (int i = 0; i < local_result->size; ++i) {
        for (int j = 0; j < local_result->size; ++j) {
            result->grid[i][j] = local_result->grid[i][j];
        }
    }
}

void ParallelBacktrackingMultiBlocksSolver::cleanup(Sudoku* local_result) {
    for (int i = 0; i < local_result->size; ++i) {
        delete[] local_result->grid[i];
    }
    delete[] local_result->grid;
    delete local_result;
}
void ParallelBacktrackingMultiBlocksSolver::display() const {

    #pragma omp parallel for
    for (int i = 0; i < result->size; ++i) {
        for (int j = 0; j < result->size; ++j) {
            std::cout << static_cast<int>(result->grid[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}