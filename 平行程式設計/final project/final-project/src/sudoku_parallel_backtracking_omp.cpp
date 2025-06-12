#include <iostream>
#include <cmath>
#include <omp.h>
#include "sudoku_parallel_backtracking.h"
#include "sudoku_serial_backtracking.h"
// #define NUM_THREADS 4


void OMPParallelBacktrackingSolver::init(const Sudoku& sudoku) {
    // omp_set_dynamic(0);
    // omp_set_num_threads(NUM_THREADS);
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    // std::cout << "Total threads: " << num_threads << '\n';
    result = new Sudoku();
    result->size = sudoku.size;
    result->grid = new uint8_t*[sudoku.size];
    for (int i = 0; i < sudoku.size; ++i) {
        result->grid[i] = new uint8_t[sudoku.size];
        for (int j = 0; j < sudoku.size; ++j) {
            result->grid[i][j] = sudoku.grid[i][j];
        }
    }
    int board_count = 0;
    bool *done = new bool(false);
    std::vector<Sudoku*> initial_choices = generate_initial_choices(0, result);
    for (Sudoku* choice : initial_choices) {
        SerialBacktrackingSolverForParallel* solver = new SerialBacktrackingSolverForParallel(*choice);
        solver->solved = done;
        solver->this_solver = false;
        solvers.push_back(solver);
        board_count++;
        
        // choice->print();
        // std::cout << "----------------\n";
    }
    // std::cout << "Total board count: " << board_count << '\n';
}

std::vector<Sudoku*> OMPParallelBacktrackingSolver::generate_initial_choices(int current_strap, const Sudoku* local_result) {
    std::vector<Sudoku*> initial_choices;
    if (current_strap == bootstrap) {
        initial_choices.push_back(new Sudoku(*local_result));
        return initial_choices;
    }

    int row, col;
    if (!find_empty(row, col, local_result)) {
        return initial_choices;
    }
    for (int num = 1; num <= local_result->size; ++num) {
        if (is_valid(row, col, num, local_result)) {
            // std::cout << "row, col, num\n" << row << ", " << col << ", " << num << '\n';
            Sudoku* choice = new Sudoku();
            choice->size = local_result->size;
            choice->grid = new uint8_t*[local_result->size];
            for (int i = 0; i < local_result->size; ++i) {
                choice->grid[i] = new uint8_t[local_result->size];
                std::copy(local_result->grid[i], 
                         local_result->grid[i] + local_result->size,
                         choice->grid[i]);
            }
            choice->grid[row][col] = num;
            // std::cout << ">>> [Before recursion] Current at strap level " << current_strap << "\n";
            std::vector<Sudoku*> next_choices = generate_initial_choices(current_strap + 1, choice);
            // std::cout << "#### Current at strap level " << current_strap << "\n";
            if (next_choices.empty()){
                // std::cout << "next choices is empty!\n";
                continue;
            }
            for (Sudoku* next_choice : next_choices) {
                initial_choices.push_back(next_choice);
                // next_choice->print();
                // std::cout << "----------------\n";
            }
        }
    }
    return initial_choices;
}

void OMPParallelBacktrackingSolver::solve() {
    
    // Find the first empty cell
    // if (!find_empty(row, col)) {
    //     return;  // No empty cells, puzzle is already solved
    // }

    int idx = -1;
    #pragma omp parallel for
    for (size_t num = 0; num < solvers.size(); ++num) {
        if (idx == -1) {
            bool found = solvers[num]->solve2();
            if (found) {
                idx = num;
            }
        }
    }
    for (size_t i = 0; i < solvers.size(); ++i) {
        if (solvers[i]->this_solver) {
            copy_result(solvers[i]->result);
            break;
        }
    }
}

bool OMPParallelBacktrackingSolver::backtracking(Sudoku* local_result) {
    int row, col;

    // Find the next empty cell
    if (!find_empty(row, col, local_result)) {
        return true;  // No empty cells left, puzzle is solved
    }

    for (int num = 1; num <= 9; ++num) {
        if (is_valid(row, col, num, local_result)) {
            local_result->grid[row][col] = num;

            if (backtracking(local_result)) {
                return true;
            }

            // Backtrack
            local_result->grid[row][col] = 0;
        }
    }
    return false;
}

bool OMPParallelBacktrackingSolver::is_valid(int row, int col, int num, const Sudoku* local_result) const {
    // Check row and column for conflicts
    for (int i = 0; i < local_result->size; ++i) {
        if (local_result->grid[row][i] == num || local_result->grid[i][col] == num) {
            return false;
        }
    }
    int grid_length = sqrt(local_result->size);
    int startRow = (row / grid_length) * grid_length;
    int startCol = (col / grid_length) * grid_length;
    
    for (int i = 0; i < grid_length; ++i) {
        for (int j = 0; j < grid_length; ++j) {
            if (local_result->grid[startRow + i][startCol + j] == num) {
                return false;
            }
        }
    }
    return true;
}

bool OMPParallelBacktrackingSolver::find_empty(int &row, int &col, const Sudoku* local_result) const {
    for (row = 0; row < local_result->size; ++row) {
        for (col = 0; col < local_result->size; ++col) {
            if (local_result->grid[row][col] == 0) {
                return true;
            }
        }
    }
    return false;
}

void OMPParallelBacktrackingSolver::copy_result(const Sudoku* local_result) {
    for (int i = 0; i < local_result->size; ++i) {
        for (int j = 0; j < local_result->size; ++j) {
            result->grid[i][j] = local_result->grid[i][j];
        }
    }
}

void OMPParallelBacktrackingSolver::cleanup(Sudoku* local_result) {
    for (int i = 0; i < local_result->size; ++i) {
        delete[] local_result->grid[i];
    }
    delete[] local_result->grid;
    delete local_result;
}