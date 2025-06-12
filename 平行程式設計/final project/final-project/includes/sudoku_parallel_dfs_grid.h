#ifndef SUDOKUPARALLELBACKTRACKINGGRIDSOLVER_H
#define SUDOKUPARALLELBACKTRACKINGGRIDSOLVER_H

#include <vector>
#include <stack>
#include <cstring>
#include "sudoku.h"
#include "sudoku_solver.h"
#include "sudoku_serial_backtracking.h"

struct Step{
    int row;
    int col;
    int number;
};

class ParallelBacktrackingGridSolver: public SudokuSolver{
public:
    // ParallelBacktrackingSolver();
    ParallelBacktrackingGridSolver(const Sudoku& sudoku){
        init(sudoku);
    }
   
    void init(const Sudoku& sudoku) override;

    void solve() override;

    void display() const;
    void parallel_pthread_backtracking();
    void parallel_mpi_backtracking();
private:
    bool find_empty(int &row, int &col, const Sudoku& local_result) const;
    void copy_result(const Sudoku& local_result);
    // std::stack<Sudoku*> board_status; // Stack to store Sudoku states
    struct ValidCheckData{
        int row;
        int col;
        int num;
        const Sudoku* sudoku;
        pthread_mutex_t* stack_mutex;
        std::stack<Sudoku*> *board_status;
        bool is_valid;
    };
    static void* solve_grid_number(void* arg);
};

#endif