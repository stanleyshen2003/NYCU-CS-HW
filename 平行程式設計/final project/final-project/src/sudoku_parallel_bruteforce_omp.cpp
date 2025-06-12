#include <iostream>
#include <cmath>
#include <omp.h>
#include "sudoku_parallel_bruteforce.h"
#define NUM_THREADS 4

void OMPParallelBruteForceSolver::init(const Sudoku& sudoku) {
    if (result != nullptr) {
        delete result;
        result = nullptr;
    }
    unsolvedBoards.clear();

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);
    
    result = new Sudoku(sudoku);

    std::vector<Sudoku*> initial_choices = init_unsolved_boards(0, result);
    //unsolvedBoards.reserve(initial_choices.size());
    for (Sudoku* choice : initial_choices) {
        SerialBruteforceSolverForParallel* sbf = new SerialBruteforceSolverForParallel(*choice);
        unsolvedBoards.push_back(sbf);
        delete choice;
    }    
}

std::vector<Sudoku*> OMPParallelBruteForceSolver::init_unsolved_boards(int current_strap, const Sudoku* local_result) {
    std::vector<Sudoku*> initial_choices;
    if (current_strap == bootstrap) {
        initial_choices.push_back(new Sudoku(*local_result));
        return initial_choices;
    }

    int row = 0;
    int col = 0;
    if (!find_empty(row, col, local_result)) {
        return initial_choices;
    }
    for (int num = 1; num <= local_result->size; ++num) {
        if (is_valid(row, col, num, local_result)) {
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
            
            std::vector<Sudoku*> next_choices = init_unsolved_boards(current_strap + 1, choice);
            if (next_choices.empty())
                continue;
            for (Sudoku* next_choice : next_choices) {
                initial_choices.push_back(next_choice);
            }
        }
    }
    return initial_choices;
}

void OMPParallelBruteForceSolver::solve() {
    int idx = -1;
    //std::cout << unsolvedBoards.size() << std::endl;

    #pragma omp parallel for shared(idx)
    for (size_t num = 0; num < unsolvedBoards.size(); ++num) {
        bool found = unsolvedBoards[num]->solve2();
        if (found) {
            #pragma omp critical
            {
                idx = num;
            }
        }
    }

    if (idx != -1) {
        copy_result(*(unsolvedBoards[idx]->result));
    }
    else {
        std::cout << "No solution found" << std::endl;
    }
}

bool OMPParallelBruteForceSolver::is_valid(int row, int col, int num, const Sudoku* sudoku) const {
    // Check row and column for conflicts
    for (int i = 0; i < sudoku->size; ++i) {
        if (sudoku->grid[row][i] == num || sudoku->grid[i][col] == num) {
            return false;
        }
    }

    // Check the subgrid for conflicts
    int grid_length = sqrt(sudoku->size);
    int startRow = (row / grid_length) * grid_length;
    int startCol = (col / grid_length) * grid_length;

    for (int i = 0; i < grid_length; ++i) {
        for (int j = 0; j < grid_length; ++j) {
            if (sudoku->grid[startRow + i][startCol + j] == num) {
                return false;
            }
        }
    }
    return true;
}

void OMPParallelBruteForceSolver::copy_result(const Sudoku& sudoku) {
    for (int i = 0; i < sudoku.size; ++i) {
        for (int j = 0; j < sudoku.size; ++j) {
            result->grid[i][j] = sudoku.grid[i][j];
        }
    }
}

bool OMPParallelBruteForceSolver::find_empty(int &row, int &col, const Sudoku* local_result) const {
     // First check the remaining cells in the current row
    for (int c = col; c < local_result->size; ++c) {
        if (local_result->grid[row][c] == 0) {
            col = c;
            return true;
        }
    }
    
    // Then check subsequent rows
    for (int r = row + 1; r < local_result->size; ++r) {
        for (int c = 0; c < local_result->size; ++c) {
            if (local_result->grid[r][c] == 0) {
                row = r;
                col = c;
                return true;
            }
        }
    }
    return false;
}