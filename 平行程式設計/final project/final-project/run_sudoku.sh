echo "---------Backtracking Serial----------";
srun ./sudoku_main --file medium --algorithm 1;

echo "---------BruteForce Serial----------";
srun ./sudoku_main --file medium --algorithm 2;

echo "---------Backtracking Openmp----------";
srun -c 1 ./sudoku_main --file medium --algorithm 4;
srun -c 2 ./sudoku_main --file medium --algorithm 4;
srun -c 3 ./sudoku_main --file medium --algorithm 4;
srun -c 4 ./sudoku_main --file medium --algorithm 4;
srun -c 5 ./sudoku_main --file medium --algorithm 4;
srun -c 6 ./sudoku_main --file medium --algorithm 4;

echo "---------Backtracking Pthread----------";
srun -c 1 ./sudoku_main --file medium --algorithm 5;
srun -c 2 ./sudoku_main --file medium --algorithm 5;
srun -c 3 ./sudoku_main --file medium --algorithm 5;
srun -c 4 ./sudoku_main --file medium --algorithm 5;
srun -c 5 ./sudoku_main --file medium --algorithm 5;
srun -c 6 ./sudoku_main --file medium --algorithm 5;

echo "---------Brute Force Openmp----------";
srun -c 1 ./sudoku_main --file medium --algorithm 6;
srun -c 2 ./sudoku_main --file medium --algorithm 6;
srun -c 3 ./sudoku_main --file medium --algorithm 6;
srun -c 4 ./sudoku_main --file medium --algorithm 6;
srun -c 5 ./sudoku_main --file medium --algorithm 6;
srun -c 6 ./sudoku_main --file medium --algorithm 6;

echo "---------Brute Force Pthread----------";
srun -c 1 ./sudoku_main --file medium --algorithm 7;
srun -c 2 ./sudoku_main --file medium --algorithm 7;
srun -c 3 ./sudoku_main --file medium --algorithm 7;
srun -c 4 ./sudoku_main --file medium --algorithm 7;
srun -c 5 ./sudoku_main --file medium --algorithm 7;
srun -c 6 ./sudoku_main --file medium --algorithm 7;
