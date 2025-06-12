#ifndef SUDOKU_PARALLEL_BACKTRACKING_MPI_H
#define SUDOKU_PARALLEL_BACKTRACKING_MPI_H

#include "sudoku.h"
#include "sudoku_solver.h"
#include <vector>
#include <mpi.h>

class MPIBacktrackingSolver: public SudokuSolver{
private:
    int rank;           // Current process rank
    int world_size;     // Total number of processes
    int bootstrap = 4;  // Initial branching depth
    std::vector<Sudoku*> local_boards;

public:
    MPIBacktrackingSolver();
    MPIBacktrackingSolver(const Sudoku& sudoku);
    ~MPIBacktrackingSolver() override;
   
    void init(const Sudoku& sudoku) override;
    void solve() override;

private:
    std::vector<Sudoku*> generate_initial_boards();
    void distribute_work(const std::vector<Sudoku*>& boards);
    void receive_work();
    void copy_to_result(const Sudoku& source);
    bool is_valid(int row, int col, int num, const Sudoku* sudoku) const;
};


#endif