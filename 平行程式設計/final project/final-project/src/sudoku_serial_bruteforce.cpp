#include <iostream>
#include <cmath>
#include <vector>
#include "sudoku_serial_bruteforce.h"
using namespace std;

void SerialBruteForceSolver::init(const Sudoku& sudoku) {
    result = new Sudoku();
    result->copyFrom(sudoku);
}

void SerialBruteForceSolver::solve() {
    int row = 0;
    int col = 0;

    std::queue<vector<State>> q; // state: previous find row and column and number
    Sudoku* blank_sudoku = new Sudoku();
    blank_sudoku->copyFrom(*result);
    vector<pair<int, int>> empty_cells;
    for(int i = 0; i < result->size; i++){
        for(int j = 0; j < result->size; j++){
            if(result->grid[i][j] == 0){
                empty_cells.push_back(make_pair(i, j));
            }
        }
    }
    //= new Sudoku(result->size, result->grid);
    // q.push(std::make_pair(row, col));
    // push first cell state
    bool is_solved = !find_empty(row, col, *blank_sudoku);
    //std::cout << row << " " << col << "\n";
    if (!is_solved) {
        for (uint8_t num = 1; num <= result->size; num++) {
            if (is_valid(row, col, num, *blank_sudoku))
                q.push({{num}});
        }
    }
    while (!q.empty() && !is_solved) {
        //std::cout << "In while...\n";
        // std::pair<int, int> current = q.front();
        vector<State> current = q.front();
        q.pop();
        // std::cout << "Current size: " << current.size() << '\n';
        for (size_t i = 0; i < current.size(); i++)
            blank_sudoku->grid[empty_cells[i].first][empty_cells[i].second] = current[i].num;
        // std::cout <<"(row, col): " << row << ", " << col << "\n";
        // result->print();
        row = empty_cells[current.size() - 1].first;
        col = empty_cells[current.size() - 1].second;
        if (!find_empty(row, col, *blank_sudoku)) {  // auto find the new empty row, col
            is_solved = true;
            break;
        } 
        // current.sudoku->print();
        for(uint8_t num = 1; num <= result->size; num++){

            if(is_valid(row, col, num, *blank_sudoku)){
                current.push_back({num});
                q.push(current);
                current.pop_back();
            }
        }
        //blank_sudoku->print();
        for (size_t i = 0; i < empty_cells.size(); i++)
            blank_sudoku->grid[empty_cells[i].first][empty_cells[i].second] = 0;
    }
    // if sudoku is solved, then free all sudoku elements in queue
    // avoid doesn't free memory because of break
    while (!q.empty()) {
        vector<State> current = q.front();
        q.pop();
        current.clear();
    }
    for (int i = 0;i<result->size;i++){
        for(int j = 0;j<result->size;j++){
            result->grid[i][j] = blank_sudoku->grid[i][j];
        }
    }
    delete blank_sudoku;
}

bool SerialBruteForceSolver::is_valid(int row, int col, int num, Sudoku& ref) const {
    // Check row and column for conflicts
    for (int i = 0; i < ref.size; ++i) {
        if (ref.grid[row][i] == num || ref.grid[i][col] == num) {
            return false;
        }
    }

    // Check the subgrid for conflicts
    int grid_length = sqrt(ref.size);
    int startRow = (row / grid_length) * grid_length;
    int startCol = (col / grid_length) * grid_length;

    for (int i = 0; i < grid_length; ++i) {
        for (int j = 0; j < grid_length; ++j) {
            if (ref.grid[startRow + i][startCol + j] == num) {
                return false;
            }
        }
    }
    return true;
}

void SerialBruteForceSolver::display() const {
    for (int i = 0; i < result->size; ++i) {
        for (int j = 0; j < result->size; ++j) {
            std::cout << static_cast<int>(result->grid[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}

bool SerialBruteForceSolver::find_empty(int &row, int &col, Sudoku& s) const {
     // First check the remaining cells in the current row
    for (int c = col; c < s.size; ++c) {
        if (s.grid[row][c] == 0) {
            col = c;
            return true;
        }
    }
    
    // Then check subsequent rows
    for (int r = row + 1; r < s.size; ++r) {
        for (int c = 0; c < s.size; ++c) {
            if (s.grid[r][c] == 0) {
                row = r;
                col = c;
                return true;
            }
        }
    }
    return false;
}