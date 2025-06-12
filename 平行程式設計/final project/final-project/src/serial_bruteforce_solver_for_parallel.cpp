#include <queue>
#include "sudoku_parallel_bruteforce.h"

bool SerialBruteforceSolverForParallel::solve2() {
    int row = 0;
    int col = 0;

    std::queue<std::vector<int>> q; // state: previous find row and column and number
    Sudoku blank_sudoku;
    blank_sudoku.copyFrom(*result);

    std::vector<std::pair<int, int>> empty_cells;
    for(int i = 0; i < result->size; i++){
        for(int j = 0; j < result->size; j++){
            if(result->grid[i][j] == 0){
                empty_cells.push_back(std::make_pair(i, j));
            }
        }
    }
    //= new Sudoku(result->size, result->grid);
    // q.push(std::make_pair(row, col));
    // push first cell state
    bool is_solved = !find_empty(row, col, blank_sudoku);
    //std::cout << row << " " << col << "\n";
    if (!is_solved) {
        for (uint8_t num = 1; num <= result->size; num++) {
            if (is_valid(row, col, num, blank_sudoku))
                q.push({num});
        }
    }
    while (!q.empty() && !is_solved) {
        //std::cout << "In while...\n";
        // std::pair<int, int> current = q.front();
        std::vector<int> current = q.front();
        q.pop();
        // std::cout << "Current size: " << current.size() << '\n';

        for (size_t i = 0; i < current.size(); i++)
            blank_sudoku.grid[empty_cells[i].first][empty_cells[i].second] = current[i];
        // std::cout <<"(row, col): " << row << ", " << col << "\n";
        // result->print();
        row = empty_cells[current.size() - 1].first;
        col = empty_cells[current.size() - 1].second;
        if (!find_empty(row, col, blank_sudoku)) {  // auto find the new empty row, col
            is_solved = true;
            break;
        } 
        // current.sudoku->print();
        for(uint8_t num = 1; num <= result->size; num++){

            if(is_valid(row, col, num, blank_sudoku)){
                current.push_back(num);
                q.push(current);
                current.pop_back();
            }
        }
        //blank_sudoku.print();
        for (size_t i = 0; i < current.size(); i++)
            blank_sudoku.grid[empty_cells[i].first][empty_cells[i].second] = 0;
    }
    // if sudoku is solved, then free all sudoku elements in queue
    // avoid doesn't free memory because of break
    while (!q.empty())
        q.pop();
    
    if (is_solved) {
        for (int i = 0;i<result->size;i++){
            for(int j = 0;j<result->size;j++){
                result->grid[i][j] = blank_sudoku.grid[i][j];
            }
        }
    }
    return is_solved;
}
