#include "sudoku_serial_genetic.h"
#include <set>
#include <map>

#include <algorithm>
#include <random>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <iostream>


void Candidate::update_fitness() {
    std::vector<int> column_count(sudoku.size, 0);
    std::vector<int> block_count(sudoku.size, 0);
    double column_sum = 0.0;
    double block_sum = 0.0;

    // Calculate column fitness
    for (int i = 0; i < sudoku.size; ++i) {
        for (int j = 0; j < sudoku.size; ++j) {
            column_count[sudoku.grid[j][i] - 1]++;
        }
        for (int j = 0; j < sudoku.size; ++j) {
            if (column_count[j] > 1) {
                column_sum += 1.0; /// (column_count[j]);

            }
        }
        std::fill(column_count.begin(), column_count.end(), 0);
    }

    // Calculate block fitness
    for (int i = 0; i < sudoku.size; i += 3) {
        for (int j = 0; j < sudoku.size; j += 3) {
            for (int x = 0; x < 3; ++x) {
                for (int y = 0; y < 3; ++y) {
                    block_count[sudoku.grid[i + x][j + y] - 1]++;
                }
            }
            for (int x = 0; x < sudoku.size; ++x) {
                if (block_count[x] > 1) {
                    block_sum += 1.0;// / (block_count[x]);
                }
            }
            std::fill(block_count.begin(), block_count.end(), 0);
        }
    }

    fitness = - column_sum - block_sum;

}

void Candidate::mutate(int mutate_grids, const std::vector<std::vector<uint8_t>>& given) {
    unsigned int seed = 42;
    double r;
    int real_mutate_grids = (rand() % (mutate_grids-1)) + 1;
    // std::cout << "Mutating\n";
    // sudoku.print();
    // std::cout << "Mutating " << real_mutate_grids << " grids\n";
    for (int i = 0; i < real_mutate_grids; ++i) {
        int row = rand_r(&seed) % sudoku.size;
        while (given[row].size() == 0 || given[row].size() == 1) {
            row = rand_r(&seed) % sudoku.size;
        }
        int column1 = rand_r(&seed) % given[row].size();
        int column2 = rand_r(&seed) % given[row].size();
        while (column1 == column2) {
            column2 = rand_r(&seed) % given[row].size();
        }
        column1 = given[row][column1];
        column2 = given[row][column2];
        uint8_t temp = sudoku.grid[row][column1];
        sudoku.grid[row][column1] = sudoku.grid[row][column2];
        sudoku.grid[row][column2] = temp;
    }
    // sudoku.print();


}

void Candidate::crossover(const Candidate& parent1, double crossover_portion) {
    unsigned int seed = 42;
    double r;

    for (int i = 0; i < sudoku.size; ++i) {
        r = (double)rand_r(&seed) / RAND_MAX;
        if (r < crossover_portion) {
            memcpy(sudoku.grid[i], parent1.sudoku.grid[i], sudoku.size * sizeof(uint8_t));
        }
    }
}

void Candidate::initialize() {
    // std::cout << "Initializing candidate\n";
    // std::cout << sudoku.size;
    std::vector<bool> row(sudoku.size);
    std::vector<uint8_t> row_miss;
    for (int i = 0; i < sudoku.size; ++i) {
        // std::cout << "Initializing row " << i << "\n";
        std::fill(row.begin(), row.end(), true);
        // std::cout << "Row filled\n";
        for (int j = 0; j < sudoku.size; ++j) {
            if (sudoku.grid[i][j] != 0){
                row[sudoku.grid[i][j] - 1] = false;
            }
        }
        // std::cout << "Row initialized\n";
        for (uint8_t j = 0; j < sudoku.size; ++j) {

            if (row[j]) {
                row_miss.push_back(j+1);
            }
        }
        std::random_shuffle(row_miss.begin(), row_miss.end());
        for (int j = 0; j < sudoku.size; ++j) {
            if (sudoku.grid[i][j] == 0) {
                sudoku.grid[i][j] = row_miss.back();
                row_miss.pop_back();
            }
        }
    }
    // sudoku.print();
}

Population::Population(int population_size, const Sudoku& sudoku) {
    // std::cout << "Population constructor\n";
    this->population_size = population_size;
    for (int i = 0; i < population_size; ++i) {
        // std::cout << "Creating candidate " << i << "\n";
        Candidate* candidate = new Candidate(sudoku);
        // std::cout << "Candidate created\n";
        candidate->initialize();
        // std::cout << "Candidate initialized\n";
        population.push_back(candidate);
        // std::cout << "Candidate added to population\n";
        filled_indices.push_back(i);
    }
    std::cout << "Population initialized\n";
}

void Population::print_empty_indices(){
    
    std::sort(empty_indices.begin(), empty_indices.end());
    for (auto i: empty_indices){
        std::cout << i << " ";
    }
    std::cout << "\n";
}

void Population::print_filled_indeces(){
    
    for (auto i: filled_indices){
        std::cout << i << " ";
    }
    std::cout << "\n";
}

void Population::print_vector_statistics(const std::vector<int>& vc){
    std::map<int, int> statistics;
    for (auto i: vc){
        statistics[population[i]->fitness]++;
    }
    for (auto pii: statistics){
        std::cout << pii.first << ": " << pii.second << "\n";
    }

}

void Population::print_fitness_statistics(){
    std::map<int, int> statistics;
    for (auto candidate: population){
        statistics[candidate->fitness]++;
    }
    for (auto pii: statistics){
        std::cout << pii.first << ": " << pii.second << "\n";
    }

}

void Population::selection(int selection_size) {
    while (filled_indices.size() > selection_size) {
        int index1 = 0, index2 = 0;
        while (index1 == index2) {
            index1 = rand() % filled_indices.size();
            index2 = rand() % filled_indices.size();
        }
        if (population[filled_indices[index1]]->fitness < population[filled_indices[index2]]->fitness) {
            empty_indices.push_back(filled_indices[index1]);
            filled_indices.erase(filled_indices.begin() + index1);
        } else {
            empty_indices.push_back(filled_indices[index2]);
            filled_indices.erase(filled_indices.begin() + index2);
        }
    }
}

void Population::crossover(int crossover_amount, double crossover_portion){
    for (int i = 0; i < crossover_amount; i++){
        int index1 = 0, index2 = 1;
        while (index1 == index2) {
            index1 = rand() % filled_indices.size();
            index2 = rand() % filled_indices.size();
        }
        int fill_index = empty_indices.back();
        empty_indices.pop_back();
        population[fill_index]->sudoku.copyFrom(population[filled_indices[index1]]->sudoku);
        population[fill_index]->crossover(*population[filled_indices[index2]], crossover_portion);
        filled_indices.push_back(fill_index);

    }
}

void Population::mutate(int mutate_amount, int mutate_grids, const std::vector<std::vector<uint8_t>>& given) {
    std::cout << "mutating\n";
    for (int i = 0; i < mutate_amount; i++) {
        int index = rand() % filled_indices.size();
        int fill_index = empty_indices.back();
        
        // std::cout << "from " << index << " to " << fill_index << "\n";
        empty_indices.pop_back();
        population[fill_index]->sudoku.copyFrom(population[filled_indices[index]]->sudoku);
        population[fill_index]->mutate(mutate_grids, given);
        filled_indices.push_back(fill_index);

    }
}

void Population::evolve(int selection_size, int crossover_amount, double crossover_portion, int mutate_amount, int mutate_grids, const std::vector<std::vector<uint8_t>>& given) {
    // for (int i = 0; i < population_size; ++i) {
    //     population[i]->update_fitness();
    //     if (population[i]->fitness > best_fitness) {
    //         best_fitness = population[i]->fitness;
    //         best_index = i;
    //     }
    // }
    // print_fitness_statistics();
    // std::cout << "Fitness updated\n";
    selection(selection_size);
    // print_filled_indeces();
    // print_empty_indices();
    // std::cout << "filled:\n";
    // print_vector_statistics(filled_indices);
    // std::cout << "empty:\n";
    // print_vector_statistics(empty_indices);
    // std::cout << "Mutation done\n";
    crossover(crossover_amount, crossover_portion);
    // std::cout << "Selection done\n";
    mutate(mutate_amount, mutate_grids, given);
    std::cout << "Mutate done\n";
    // std::cout << "Crossover done\n";
    for (int i = 0; i < population_size; ++i) {
        population[i]->update_fitness();
        if (population[i]->fitness > best_fitness) {
            best_fitness = population[i]->fitness;
            best_index = i;
        }
    }

    generation++;
}


SerialGeneticSolver::SerialGeneticSolver(Sudoku& sudoku) {
    std::cout << "SerialGeneticSolver constructor\n";
    sudoku.random_empty_cells(empty_count);
    sudoku.print();
    for (int i = 0; i < sudoku.size; ++i) {
        std::vector<uint8_t> row;
        std::vector<bool> row_miss(sudoku.size);
        std::fill(row_miss.begin(), row_miss.end(), true);
        
        for (int j = 0; j < sudoku.size; ++j) {
            if (sudoku.grid[i][j] != 0)
                row_miss[sudoku.grid[i][j] - 1] = false;
        }
        for (uint8_t j = 0; j < sudoku.size; ++j) {
            if (row_miss[j]) {
                row.push_back(j+1);
            }
        }
        given.push_back(row);
    }
    // std::cout << "Given initialized\n";
    population = new Population(population_size, sudoku);
    // std::cout << "Population initialized\n";
}

Sudoku SerialGeneticSolver::solve() {
    while (true) {
        std::cout << "Generation: " << population->generation << "\n";
        population->evolve(selection_size, crossover_amount, crossover_portion, mutate_amount, mutate_grids, given);
        std::cout << "Evolved\n";
        if (population->population[population->best_index]->sudoku.isValid()) {
            break;
        }
        // population->population[population->best_index]->sudoku.print();

        std::cout << "Fitness: " << population->best_fitness << "\n";
        // if (population->generation == 3)
        //     break;
        // if (population->generation > 10) {
        //     break;
        // }
    }
    Sudoku result;
    result.copyFrom(population->population[population->best_index]->sudoku);
    return result;
}

