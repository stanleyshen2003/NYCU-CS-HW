#ifndef GENETIC_SERIAL_H
#define GENETIC_SERIAL_H

#include <vector>
#include <list>
#include "sudoku.h"

class Candidate {
public:
    Sudoku sudoku;

    double fitness = 0;
    // Candidate(Sudoku sudoku){
    //     this->sudoku = Sudoku(sudoku.size, sudoku.grid);
    // };
    Candidate(const Sudoku& sudoku) : sudoku(sudoku.size, sudoku.grid) {}


    Candidate(const Candidate& other) {
        this->sudoku = Sudoku(other.sudoku.size, other.sudoku.grid);
        this->fitness = other.fitness;
    }
    
    void update_fitness();
    void mutate(int mutate_grids, const std::vector<std::vector<uint8_t>>& given);
    void crossover(const Candidate& parent1, double crossover_portion);
    void initialize();

    // = assignment operator
    Candidate& operator=(const Candidate& other) {
        if (this != &other) {
            sudoku.copyFrom(other.sudoku);
            fitness = other.fitness;
        }
        return *this;
    }

};

class Population {
public:
    std::vector<Candidate*> population;
    std::vector<int> empty_indices;
    std::vector<int> filled_indices;
    int population_size;
    int generation = 0;
    int best_index = 0;
    int best_fitness = -10000;

    Population(int population_size, const Sudoku& sudoku);

    ~Population() {
        for (Candidate* candidate : population) {
            delete candidate;
        }
    }

    void evolve(int selection_size, int crossover_amount, double crossover_portion, int mutate_amount, int mutate_grids, const std::vector<std::vector<uint8_t>>& given);
    void selection(int selection_size);
    void print_empty_indices();
    void print_fitness_statistics();
    void print_filled_indeces();
    void print_vector_statistics(const std::vector<int>& vc);

    void crossover(int crossover_amount, double crossover_portion);

    void mutate(int mutate_amount, int mutate_grids, const std::vector<std::vector<uint8_t>>& given);


};


class SerialGeneticSolver {
public:
    std::vector<std::vector<uint8_t>> given;
    Population* population;
    int population_size = 150;
    int selection_size = 100;
    int crossover_amount = 25;
    double crossover_portion = 0.5;
    int mutate_amount = 25;
    int mutate_grids = 3;
    int empty_count = 15;

    SerialGeneticSolver(Sudoku& sudoku);

    ~SerialGeneticSolver() {
        delete population;
    }

    Sudoku solve();

};


#endif
