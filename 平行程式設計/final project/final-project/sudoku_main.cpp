#include "sudoku.h"
#include "sudoku_serial_backtracking.h"
#include "sudoku_serial_bruteforce.h"
#include "sudoku_serial_genetic.h"

#include "sudoku_parallel_backtracking.h"
// #include "sudoku_parallel_backtracking_multiblocks.h"
// #include "sudoku_parallel_backtracking_mpi.h"
#include "sudoku_parallel_bruteforce.h"
#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include "CycleTimer.h"
#define CASES 10

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <path_to_sudoku_file> [--algorithm <number>]\n"
              << "Available algorithms:\n"
              << "  1  - Serial implementation using backtracking\n"
              << "  2  - Serial implementation using brute force\n"
              << "  3  - Serial implementation using genetic algorithm (deprecated)\n"
              << "  4  - Parallel implementation using backtracking algorithm\n"
              << "  5  - Parallel Pthread implementation using backtracking algorithm\n"
              << "  6  - Parallel OMP implementation using brute force algorithm\n"
              << "  7  - Parallel Pthread implementation using brute force algorithm\n"
              << "  8  - Parallel Pthread implementation using backtracking algorithm (Not worked)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Default values
    std::string filename;
    int algorithmChoice = 1; // default serial_backtracking

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--algorithm") == 0) {
            if (i + 1 < argc) {
                algorithmChoice = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: --algorithm requires a value\n";
                printUsage(argv[0]);
                return 1;
            }
        } 
        else if (strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) {
                filename = argv[++i];
            } else {
                std::cerr << "Error: --file requires a value\n";
                printUsage(argv[0]);
                return 1;
            }
        }
    }

    std::vector<std::string> files;
    if (filename == "medium") {
        for (int i = 1; i <= CASES; i++){
            files.push_back("mazes/16x16_Medium_" + std::to_string(i) + ".txt");
        }
    }
    else{
        files.push_back(filename);
    }

    if (filename.empty()) {
        std::cerr << "Error: No input file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    std::string algorithmName;

    double startTime = CycleTimer::currentSeconds();
    std::vector<double> times;
    for (auto file : files){
        // Load the Sudoku puzzle
        double start = CycleTimer::currentSeconds();
        Sudoku sudoku;
        sudoku.loadSudoku(file);
        std::unique_ptr<SudokuSolver> solver;
        

    // Create the appropriate solver based on the algorithm choice
    switch (algorithmChoice) {
        case 1: // Serial backtracking
            solver = std::make_unique<SerialBacktrackingSolver>(sudoku);
            algorithmName = "Serial backtracking (DFS)";
            break;
        case 2: // Serial brute force
            solver = std::make_unique<SerialBruteForceSolver>(sudoku);
            algorithmName = "Serial bruteforce (BFS)";
            break;
        case 3: // Serial genetic algorithm
            // solver = std::make_unique<SerialGeneticSolver>(sudoku);
            break;
        case 4:
            solver = std::make_unique<OMPParallelBacktrackingSolver>(sudoku);
            algorithmName = "Parallel OMP backtracking (DFS)";
            break;
        case 5:
            solver = std::make_unique<PthreadParallelBacktrackingSolver>(sudoku);
            algorithmName = "Parallel Pthread backtracking (DFS)";
            break;
        case 6:
            solver = std::make_unique<OMPParallelBruteForceSolver>(sudoku);
            algorithmName = "Parallel OMP bruteforce (BFS)";
            break;
        case 7:
            solver = std::make_unique<PthreadParallelBruteForceSolver>(sudoku);
            algorithmName = "Parallel Pthread bruteforce (BFS)";
            break;
        default:
            std::cerr << "Error: Unknown algorithm number '" << algorithmChoice << "'\n";
            algorithmName = "Unknown";
            printUsage(argv[0]);
            return 1;
    }

        solver->solve();

        double end = CycleTimer::currentSeconds();
        times.push_back((end - start) * 1000);
        if (!solver->result->isValid()) {
            std::cerr << "Error: Invalid solution on " << file <<"\n";
            return 1;
        }
        
    }
    double endTime = CycleTimer::currentSeconds();
    for (size_t i = 0; i < times.size(); i++){
        std::cout << "Time for " << files[i] << ": " << times[i] << " ms\n";
    }
    std::cout << "Total Time: " << (endTime - startTime) * 1000 << " ms\n";
    
    // Print results
    std::cout << "Using algorithm: " << algorithmName << "\n";
    std::cout << "Sudoku puzzle loaded from: " << filename << "\n";
    // solver->result->print();

    // bool valid = solver->result->isValid();
    // std::cout << "Sudoku is " << (valid ? "valid." : "invalid.") << std::endl;

    return 0;
}
