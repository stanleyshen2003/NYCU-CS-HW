#include "sudoku_parallel_bruteforce.h"
#include "sudoku_parallel_bruteforce_mpi.h"
#include <mpi.h>
#include <cmath>

MPIBruteForceSolver::MPIBruteForceSolver() {
    result = nullptr;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

MPIBruteForceSolver::MPIBruteForceSolver(const Sudoku& sudoku) : MPIBruteForceSolver() {
    init(sudoku);
}

MPIBruteForceSolver::~MPIBruteForceSolver() {
    for (Sudoku* board : local_boards) {
        delete board;
    }
}

void MPIBruteForceSolver::init(const Sudoku& sudoku) {
    if (result != nullptr) {
        delete result;
    }
    result = new Sudoku(sudoku);

    if (rank == 0) {
        // Root process generates initial boards
        std::vector<Sudoku*> initial_boards = generate_initial_boards();
        distribute_work(initial_boards);

        for (Sudoku* board : initial_boards) {
            delete board;
        }
    } else {
        // Worker processes wait for their boards
        receive_work();
    }
}

void MPIBruteForceSolver::solve() {
    bool local_solution_found = false;
    Sudoku* local_solution = nullptr;

    
    // Try to solve each local board
    for (Sudoku* board : local_boards) {
        SerialBruteforceSolverForParallel* solver = new SerialBruteforceSolverForParallel(*board);
        if (solver->solve2()) {
            local_solution_found = true;
            local_solution = new Sudoku(*(solver->result));
            break;
        }
    }

    // Gather results
    if (rank == 0) {
        if (local_solution_found) {
            copy_to_result(*local_solution);
        }
        // Check solutions from other processes
        for (int i = 1; i < world_size; i++) {
            int has_solution;
            MPI_Recv(&has_solution, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (has_solution) {
                std::vector<uint8_t> grid_data(result->size * result->size);
                MPI_Recv(grid_data.data(), grid_data.size(), MPI_UINT8_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (int r = 0; r < result->size; r++) {
                    for (int c = 0; c < result->size; c++) {
                        result->grid[r][c] = grid_data[r * result->size + c];
                    }
                }
                break;  // We found a solution
            }
        }
    } else {
        // Send local solution to root
        int has_solution = local_solution_found ? 1 : 0;
        MPI_Send(&has_solution, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        if (local_solution_found) {
            std::vector<uint8_t> grid_data(local_solution->size * local_solution->size);
            for (int r = 0; r < local_solution->size; r++) {
                for (int c = 0; c < local_solution->size; c++) {
                    grid_data[r * local_solution->size + c] = local_solution->grid[r][c];
                }
            }
            MPI_Send(grid_data.data(), grid_data.size(), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);
        }
    }

    delete local_solution;
}

std::vector<Sudoku*> MPIBruteForceSolver::generate_initial_boards() {
    std::vector<Sudoku*> boards;
    std::queue<Sudoku*> q;
    q.push(new Sudoku(*result));

    int depth = 0;
    while (depth < bootstrap && !q.empty()) {
        int level_size = q.size();
        for (int i = 0; i < level_size; i++) {
            Sudoku* current = q.front();
            q.pop();

            int row = 0, col = 0;
            bool found = false;
            for (int r = 0; r < current->size; r++) {
                for (int c = 0; c < current->size; c++) {
                    if (current->grid[r][c] == 0) {
                        row = r;
                        col = c;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }

            if (!found) {
                boards.push_back(current);
                continue;
            }

            for (uint8_t num = 1; num <= current->size; num++) {
                if (is_valid(row, col, num, current)) {
                    Sudoku* next = new Sudoku(*current);
                    next->grid[row][col] = num;
                    if (depth == bootstrap - 1) {
                        boards.push_back(next);
                    } else {
                        q.push(next);
                    }
                }
            }
            delete current;
        }
        depth++;
    }

    while (!q.empty()) {
        delete q.front();
        q.pop();
    }

    return boards;
}

void MPIBruteForceSolver::distribute_work(const std::vector<Sudoku*>& boards) {
    int total_boards = boards.size();
    int base_boards_per_proc = total_boards / world_size;
    int extra_boards = total_boards % world_size;

    int start_idx = 0;
    for (int i = 0; i < world_size; i++) {
        int boards_for_this_proc = base_boards_per_proc + (i < extra_boards ? 1 : 0);
        
        if (i == 0) {
            // Keep local boards for rank 0
            for (int j = 0; j < boards_for_this_proc; j++) {
                local_boards.push_back(new Sudoku(*boards[start_idx + j]));
            }
        } else {
            // Send board count
            MPI_Send(&boards_for_this_proc, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            
            // Send each board
            for (int j = 0; j < boards_for_this_proc; j++) {
                Sudoku* board = boards[start_idx + j];
                // Send size and grid data
                std::vector<uint8_t> grid_data(board->size * board->size);
                for (int r = 0; r < board->size; r++) {
                    for (int c = 0; c < board->size; c++) {
                        grid_data[r * board->size + c] = board->grid[r][c];
                    }
                }
                MPI_Send(&board->size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(grid_data.data(), grid_data.size(), MPI_UINT8_T, i, 0, MPI_COMM_WORLD);
            }
        }
        start_idx += boards_for_this_proc;
    }
}

void MPIBruteForceSolver::receive_work() {
    int board_count;
    MPI_Recv(&board_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < board_count; i++) {
        int size;
        MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<uint8_t> grid_data(size * size);
        MPI_Recv(grid_data.data(), grid_data.size(), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        Sudoku* board = new Sudoku();
        board->size = size;
        board->grid = new uint8_t*[size];
        for (int r = 0; r < size; r++) {
            board->grid[r] = new uint8_t[size];
            for (int c = 0; c < size; c++) {
                board->grid[r][c] = grid_data[r * size + c];
            }
        }
        local_boards.push_back(board);
    }
}

void MPIBruteForceSolver::copy_to_result(const Sudoku& source) {
    for (int i = 0; i < result->size; i++) {
        for (int j = 0; j < result->size; j++) {
            result->grid[i][j] = source.grid[i][j];
        }
    }
}

bool MPIBruteForceSolver::is_valid(int row, int col, int num, const Sudoku* sudoku) const {
    // Check row
    for (int i = 0; i < sudoku->size; i++) {
        if (sudoku->grid[row][i] == num) return false;
    }
    
    // Check column
    for (int i = 0; i < sudoku->size; i++) {
        if (sudoku->grid[i][col] == num) return false;
    }
    
    // Check box
    int box_size = sqrt(sudoku->size);
    int box_row = (row / box_size) * box_size;
    int box_col = (col / box_size) * box_size;
    
    for (int i = 0; i < box_size; i++) {
        for (int j = 0; j < box_size; j++) {
            if (sudoku->grid[box_row + i][box_col + j] == num) return false;
        }
    }
    
    return true;
}