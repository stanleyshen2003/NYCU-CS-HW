#include "sudoku_parallel_backtracking.h"
#include <cmath>
#include <queue>
#include <iostream>

void PthreadParallelBacktrackingSolver::init(const Sudoku& sudoku) {
    if (result != nullptr) {
        delete result;
        result = nullptr;
    }
    unsolvedBoards.clear();
    
    result = new Sudoku(sudoku);

    std::vector<Sudoku*> initial_choices = init_unsolved_boards(0, result);
    for (Sudoku* choice : initial_choices) {
        SerialBacktrackingSolverForParallel* sbf = new SerialBacktrackingSolverForParallel(*choice);
        sbf->solved = new bool(false);
        sbf->this_solver = false;
        unsolvedBoards.push_back(sbf);
        delete choice;
    }
}

void* PthreadParallelBacktrackingSolver::solve_thread(void* arg) {
    BacktrackingThreadData* data = static_cast<BacktrackingThreadData*>(arg);
    if (*(data->solver->solved)) {
        return nullptr;
    }
    bool found = data->solver->solve2();
    
    if (found) {
        int expected = -1;
        if (data->solution_found->compare_exchange_strong(expected, data->thread_id)) {
            // First thread to find a solution
            return reinterpret_cast<void*>(1);
        }
    }
    return nullptr;
}

void PthreadParallelBacktrackingSolver::solve() {
    std::vector<pthread_t> threads(unsolvedBoards.size());
    std::vector<BacktrackingThreadData> thread_data(unsolvedBoards.size());
    std::atomic<int> solution_found{-1};

    // Create threads
    for (size_t i = 0; i < unsolvedBoards.size(); ++i) {
        thread_data[i] = {
            unsolvedBoards[i],
            &solution_found,
            static_cast<int>(i)
        };
        
        pthread_create(&threads[i], nullptr, solve_thread, &thread_data[i]);
    }

    // Wait for all threads to complete
    for (size_t i = 0; i < threads.size(); ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Check if a solution was found
    int solved_idx = solution_found.load();
    if (solved_idx != -1) {
        pthread_mutex_lock(&result_mutex);
        copy_result(*(unsolvedBoards[solved_idx]->result));
        pthread_mutex_unlock(&result_mutex);
    } else {
        std::cout << "No solution found" << std::endl;
    }
}

std::vector<Sudoku*> PthreadParallelBacktrackingSolver::init_unsolved_boards(int current_strap, const Sudoku* local_result) {
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

bool PthreadParallelBacktrackingSolver::is_valid(int row, int col, int num, const Sudoku* sudoku) const {
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

void PthreadParallelBacktrackingSolver::copy_result(const Sudoku& sudoku) {
    for (int i = 0; i < sudoku.size; ++i) {
        for (int j = 0; j < sudoku.size; ++j) {
            result->grid[i][j] = sudoku.grid[i][j];
        }
    }
}

bool PthreadParallelBacktrackingSolver::find_empty(int &row, int &col, const Sudoku* local_result) const {
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