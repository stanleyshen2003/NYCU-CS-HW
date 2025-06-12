#ifndef SUDOKUSOLVER_H
#define SUDOKUSOLVER_H
#include "sudoku.h"

class SudokuSolver{
    public:
        Sudoku* result;
        virtual void init(const Sudoku& sudoku) = 0;
        virtual void solve() = 0;
    
        virtual ~SudokuSolver() {
            if (result != nullptr) {
                //deallocateSudoku(*result);
                delete result;
                result = nullptr;
            }
        }
};
#endif