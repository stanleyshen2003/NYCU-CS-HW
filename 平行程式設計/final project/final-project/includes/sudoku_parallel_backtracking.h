#ifndef SUDOKUPARALLELBACKTRACKINGSOLVER_H
#define SUDOKUPARALLELBACKTRACKINGSOLVER_H

#include <vector>
#include "sudoku.h"
#include "sudoku_solver.h"
#include "sudoku_serial_backtracking.h"

class SerialBacktrackingSolverForParallel: public SerialBacktrackingSolver{
    public:
        bool this_solver;
        bool* solved;
        bool backtracking();
        SerialBacktrackingSolverForParallel(Sudoku sudoku): SerialBacktrackingSolver(sudoku) {this_solver = false;};
        bool solve2();
};

class OMPParallelBacktrackingSolver: public SudokuSolver{
public:
    OMPParallelBacktrackingSolver(){
        result = nullptr;
    }
    OMPParallelBacktrackingSolver(const Sudoku& sudoku){
        result = nullptr;
        init(sudoku);
    }
    ~OMPParallelBacktrackingSolver() override = default;
    void init(const Sudoku& sudoku) override;
    void solve() override;

private:
    bool find_empty(int &row, int &col, const Sudoku* local_result) const;
    bool backtracking(Sudoku* local_result);
    void copy_result(const Sudoku* local_result);
    void cleanup(Sudoku* local_result);
    bool is_valid(int row, int col, int num, const Sudoku* local_result) const;
    std::vector<Sudoku*> generate_initial_choices(int current_strap, const Sudoku* local_result);

private:
    std::vector<SerialBacktrackingSolverForParallel*> solvers;
    int bootstrap = 4;
};

/////////////// pthread version ///////////////
#include <pthread.h>
#include <atomic>

struct BacktrackingThreadData {
    SerialBacktrackingSolverForParallel* solver;
    std::atomic<int>* solution_found;
    int thread_id;
};

class PthreadParallelBacktrackingSolver: public SudokuSolver {
public:
    PthreadParallelBacktrackingSolver() : num_threads(4) {
        result = nullptr;
    }
    PthreadParallelBacktrackingSolver(const Sudoku& sudoku) : num_threads(4) {
        result = nullptr;
        init(sudoku);
    }
    ~PthreadParallelBacktrackingSolver() override {
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
    std::vector<SerialBacktrackingSolverForParallel*> unsolvedBoards;
    pthread_mutex_t result_mutex = PTHREAD_MUTEX_INITIALIZER;
};
#endif