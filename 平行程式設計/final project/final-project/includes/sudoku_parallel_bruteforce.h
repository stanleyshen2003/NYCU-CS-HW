#ifndef SUDOKUPARALLELBRUTEFORCESOLVER_H
#define SUDOKUPARALLELBRUTEFORCESOLVER_H

#include "sudoku.h"
#include "sudoku_solver.h"
#include "sudoku_serial_bruteforce.h"
#include <vector>
#include <iostream>

class SerialBruteforceSolverForParallel: public SerialBruteForceSolver{
    public:
        SerialBruteforceSolverForParallel(Sudoku sudoku) : SerialBruteForceSolver(sudoku) {};
        ~SerialBruteforceSolverForParallel() override = default;
        
        bool solve2();
        SerialBruteforceSolverForParallel(const SerialBruteforceSolverForParallel&) = delete;
        SerialBruteforceSolverForParallel& operator=(const SerialBruteforceSolverForParallel&) = delete;
        SerialBruteforceSolverForParallel(SerialBruteforceSolverForParallel&& other) noexcept 
            : SerialBruteForceSolver(std::move(other)) {
            result = other.result;
            other.result = nullptr;
        }
        SerialBruteforceSolverForParallel& operator=(SerialBruteforceSolverForParallel&&) noexcept;
};

/////////////// OpenMP version ///////////////
class OMPParallelBruteForceSolver: public SudokuSolver {
public:
    OMPParallelBruteForceSolver() {
        result = nullptr;
    }
    OMPParallelBruteForceSolver(const Sudoku& sudoku){
        result = nullptr;
        init(sudoku);
    }
    ~OMPParallelBruteForceSolver() override = default;
    void init(const Sudoku& sudoku) override;
    void solve() override;

private:
    bool is_valid(int row, int col, int num, const Sudoku* sudoku) const;
    void copy_result(const Sudoku& sudoku);
    bool find_empty(int &row, int &col, const Sudoku* local_result) const;
    std::vector<Sudoku*> init_unsolved_boards(int current_strap, const Sudoku* local_result);
    
private:
    int bootstrap = 4;
    std::vector<SerialBruteforceSolverForParallel*> unsolvedBoards;
};

/////////////// pthread version ///////////////
#include <pthread.h>
#include <atomic>

struct ThreadData {
    SerialBruteforceSolverForParallel* solver;
    std::atomic<int>* solution_found;
    int thread_id;
};

class PthreadParallelBruteForceSolver: public SudokuSolver {
public:
    PthreadParallelBruteForceSolver() : num_threads(4) {
        result = nullptr;
    }
    PthreadParallelBruteForceSolver(const Sudoku& sudoku) : num_threads(4) {
        result = nullptr;
        init(sudoku);
    }
    ~PthreadParallelBruteForceSolver() override {
        for (auto solver : unsolvedBoards) {
            delete solver;
        }
    }
    
    void init(const Sudoku& sudoku) override;
    void solve() override;

private:
    bool is_valid(int row, int col, int num, const Sudoku* sudoku) const;
    void copy_result(const Sudoku& sudoku);
    bool find_empty(int &row, int &col, const Sudoku* local_result) const;
    std::vector<Sudoku*> init_unsolved_boards(int current_strap, const Sudoku* local_result);
    static void* solve_thread(void* arg);

private:
    const int num_threads;
    int bootstrap = 4;
    std::vector<SerialBruteforceSolverForParallel*> unsolvedBoards;
    pthread_mutex_t result_mutex = PTHREAD_MUTEX_INITIALIZER;
};
#endif
