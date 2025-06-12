#include "sudoku_parallel_backtracking.h"

bool SerialBacktrackingSolverForParallel::solve2() {
    return backtracking();
}

bool SerialBacktrackingSolverForParallel::backtracking() {
    if (*solved) {
        return true;
    }

    int row, col;

    // Find the next empty cell
    if (!find_empty(row, col)) {
        *solved = true;
        this_solver = true;
        return true;  // No empty cells left, puzzle is solved
    }

    for (int num = 1; num <= result->size; ++num) {
        if (is_valid(row, col, num)) {
            result->grid[row][col] = num;

            if (backtracking()) {
                return true;
            }

            // Backtrack
            result->grid[row][col] = 0;
        }
    }
    return false;
}
