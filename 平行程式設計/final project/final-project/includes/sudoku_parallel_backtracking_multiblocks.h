#ifndef SUDOKUPARALLELBACKTRACKINGMULTIBLOCKSSOLVER_H
#define SUDOKUPARALLELBACKTRACKINGMULTIBLOCKSSOLVER_H


#include "sudoku.h"
#include "sudoku_solver.h"

class ParallelBacktrackingMultiBlocksSolver: public SudokuSolver{
public:
    ParallelBacktrackingMultiBlocksSolver();
    ParallelBacktrackingMultiBlocksSolver(const Sudoku& sudoku){
        init(sudoku);
    }

    void init(const Sudoku& sudoku) override;

    bool is_valid(int row, int col, int num) const;

    void solve() override;

    void display() const;

private:
    bool find_empty(int &row, int &col) const;
    void backtracking();
    void copy_result(const Sudoku* local_result);
    void cleanup(Sudoku* local_result);
    struct State {
        int row;  // Current row being processed
        int col;  // Current column being processed
        int num;  // Current number being tried
    };
};

#endif