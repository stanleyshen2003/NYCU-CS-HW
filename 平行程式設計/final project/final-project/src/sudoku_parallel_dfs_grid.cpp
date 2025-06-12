#include <iostream>
#include <cmath>
#include <omp.h>
#include <pthread.h>
#include "sudoku_parallel_dfs_grid.h"

// extern Sudoku* result;
bool is_valid(int row, int col, int num, const Sudoku& local_result) {
    // Check row and column for conflicts
    for (int i = 0; i < local_result.size; ++i) {
        if (local_result.grid[row][i] == num || local_result.grid[i][col] == num) {
            return false;
        }
    }
    int grid_length = sqrt(local_result.size);
    int startRow = (row / grid_length) * grid_length;
    int startCol = (col / grid_length) * grid_length;
    
    for (int i = 0; i < grid_length; ++i) {
        for (int j = 0; j < grid_length; ++j) {
            if (local_result.grid[startRow + i][startCol + j] == num) {
                return false;
            }
        }
    }
    return true;
}

void ParallelBacktrackingGridSolver::init(const Sudoku& sudoku) {
    
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


void ParallelBacktrackingGridSolver::solve() {
    
    parallel_pthread_backtracking();
    // parallel_mpi_backtracking();
}
void* ParallelBacktrackingGridSolver::solve_grid_number(void* arg){
    ValidCheckData* data = static_cast<ValidCheckData*>(arg);
    data->is_valid = is_valid(data->row, data->col, data->num, *data->sudoku);
    if(data->is_valid){
        Sudoku* new_sudoku = new Sudoku(*data->sudoku);
        new_sudoku->grid[data->row][data->col] = data->num;
        pthread_mutex_lock(data->stack_mutex);
        data->board_status->push(new_sudoku);
        pthread_mutex_unlock(data->stack_mutex);
    }
    return nullptr;
}
void ParallelBacktrackingGridSolver::parallel_pthread_backtracking() {
    int row, col;
    std::stack<Sudoku*> board_status; // Stack to store Sudoku states
    pthread_mutex_t stack_mutex = PTHREAD_MUTEX_INITIALIZER;
    board_status.push(new Sudoku(*result));
    bool solved = false;
    while(!board_status.empty() && !solved){
        Sudoku* local_result = board_status.top();
        board_status.pop();

        if(!find_empty(row, col, *local_result)){
            copy_result(*local_result);
            solved = true;
            pthread_mutex_destroy(&stack_mutex);
            break;
        }
        int max_num = local_result->size;
        pthread_t threads[max_num];
        ValidCheckData data[max_num];
        int thread_count = 0;
        for(int num = 1; num <= max_num; num++){
            data[thread_count].row = row;
            data[thread_count].col = col;
            data[thread_count].num = num;
            data[thread_count].sudoku = local_result;
            data[thread_count].stack_mutex = &stack_mutex;
            data[thread_count].board_status = &board_status;
            data[thread_count].is_valid = false;
            int ret_val = pthread_create(&threads[thread_count], nullptr, solve_grid_number, &data[thread_count]);
            if(ret_val != 0){
                std::cerr << "Error: pthread_create failed for num " << num << std::endl;
                // 處理錯誤，可能跳過這個數字
                continue;
            }
            thread_count++;
        }
        for(int i = 0; i < thread_count; i++) {
            pthread_join(threads[i], nullptr);
        }
    }
    pthread_mutex_destroy(&stack_mutex);
}
void ParallelBacktrackingGridSolver::parallel_mpi_backtracking() {

    std::stack<std::vector<Step>> board_status; // Stack to store Sudoku states
    int row, col;

    board_status.push(std::vector<Step>());
    
    while(!board_status.empty()){
        std::vector<Step> current_steps = board_status.top();
        board_status.pop();
        Sudoku* local_result = new Sudoku(*result);
        for (Step step : current_steps) {
            local_result->grid[step.row][step.col] = step.number;
        }
        // Find the first empty cell
        if (!find_empty(row, col, *local_result)) {
            copy_result(*local_result);
            return; // No empty cells, puzzle already solved
        }

        for(int num = 1; num <= local_result->size; num++) {
            if(is_valid(row, col, num, *local_result)) {
                std::vector<Step> new_steps = current_steps;
                new_steps.emplace_back(Step{row, col, num});
                board_status.push(new_steps);
            }
        }
    }
}

bool ParallelBacktrackingGridSolver::find_empty(int &row, int &col, const Sudoku& local_result) const {
    for (row = 0; row < local_result.size; ++row) {
        for (col = 0; col < local_result.size; ++col) {
            if (local_result.grid[row][col] == 0) {
                return true;
            }
        }
    }
    return false;
}

void ParallelBacktrackingGridSolver::copy_result(const Sudoku& local_result) {
    for (int i = 0; i < local_result.size; ++i) {
        std::memcpy(result->grid[i], local_result.grid[i], local_result.size * sizeof(uint8_t));
    }
}

void ParallelBacktrackingGridSolver::display() const {

    #pragma omp parallel for
    for (int i = 0; i < result->size; ++i) {
        for (int j = 0; j < result->size; ++j) {
            std::cout << static_cast<int>(result->grid[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}
