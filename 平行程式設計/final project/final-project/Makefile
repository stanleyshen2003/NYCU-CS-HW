# Makefile for Sudoku Solver

# Compilers
CXX = g++
MPICXX = mpic++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++17 -fopenmp -pthread
MPICXXFLAGS = -Wall -Wextra -std=c++17

# All output target executables
TARGETS = sudoku_main sudoku_mpi_main

COMMON_DEPENDENCIES = \
	./src/sudoku.cpp \
	./src/sudoku_serial_bruteforce.cpp \
	./src/sudoku_serial_backtracking.cpp \
	./src/sudoku_parallel_backtracking_omp.cpp \
 	./src/sudoku_parallel_backtracking_pthread.cpp \
	./src/sudoku_parallel_bruteforce_omp.cpp \
	./src/sudoku_parallel_bruteforce_pthread.cpp \
	./src/serial_bruteforce_solver_for_parallel.cpp \
	./src/serial_backtracking_solver_for_parallel.cpp \
	./src/sudoku_parallel_backtracking_multiblocks.cpp

MPI_SOURCES = \
	./src/sudoku.cpp \
	./src/sudoku_serial_bruteforce.cpp \
	./src/sudoku_serial_backtracking.cpp \
	./src/sudoku_parallel_bruteforce_mpi.cc \
	./src/sudoku_parallel_backtracking_mpi.cc \
	./src/serial_bruteforce_solver_for_parallel.cpp \
	./src/serial_backtracking_solver_for_parallel.cpp

all: $(TARGETS)

sudoku_main: sudoku_main.cpp $(COMMON_DEPENDENCIES)
	$(CXX) $(CXXFLAGS) -I ./includes -o $@ $^

sudoku_mpi_main: sudoku_mpi_main.cc $(MPI_SOURCES)
	$(MPICXX) $(MPICXXFLAGS) -I ./includes -o $@ $^

clean:
	rm -f $(TARGETS) solution.txt

.PHONY: all clean