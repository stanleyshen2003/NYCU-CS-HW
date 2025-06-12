#ifndef SUDOKU_H
#define SUDOKU_H

#include <string>
#include <cstdint>

class Sudoku {
private:
    void allocateGrid();
    void deallocateGrid();

public:
    int size;
    uint8_t** grid;

    Sudoku();
    explicit Sudoku(int size);
    Sudoku(int size, uint8_t** grid);
    Sudoku(const Sudoku& other);
    Sudoku(Sudoku&& other) noexcept;
    ~Sudoku();

    Sudoku& operator=(const Sudoku& other);
    Sudoku& operator=(Sudoku&& other) noexcept;

    void copyFrom(const Sudoku& other);
    void loadSudoku(const std::string& filename);
    void print() const;
    bool isValid() const;
    void random_empty_cells(int empty_cells);
};

#endif