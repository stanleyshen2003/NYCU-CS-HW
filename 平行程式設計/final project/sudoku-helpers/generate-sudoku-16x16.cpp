#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define GRID_SIZE 16
#define SUBGRID_SIZE 4

// Shuffle numbers randomly
void shuffleNumbers(std::vector<int>& numbers) {
    std::random_shuffle(numbers.begin(), numbers.end());
}

// Check if a number can be placed at a specific position
bool isValid(std::vector<std::vector<int>>& grid, int row, int col, int num) {
    for (int x = 0; x < GRID_SIZE; x++) {
        if (grid[row][x] == num || grid[x][col] == num ||
            grid[(row / SUBGRID_SIZE) * SUBGRID_SIZE + x / SUBGRID_SIZE]
                [(col / SUBGRID_SIZE) * SUBGRID_SIZE + x % SUBGRID_SIZE] == num) {
            return false;
        }
    }
    return true;
}

// Fill the grid using backtracking
bool fillGrid(std::vector<std::vector<int>>& grid, int row, int col) {
    if (row == GRID_SIZE - 1 && col == GRID_SIZE) {
        return true;
    }
    if (col == GRID_SIZE) {
        row++;
        col = 0;
    }
    if (grid[row][col] != 0) {
        return fillGrid(grid, row, col + 1);
    }

    // Shuffle numbers before filling
    std::vector<int> numbers(GRID_SIZE);
    for (int i = 1; i <= GRID_SIZE; ++i) {
        numbers[i - 1] = i;
    }
    shuffleNumbers(numbers);

    for (int num : numbers) {
        if (isValid(grid, row, col, num)) {
            grid[row][col] = num;
            if (fillGrid(grid, row, col + 1)) {
                return true;
            }
            grid[row][col] = 0;
        }
    }
    return false;
}

// Generate a fully solved Sudoku grid
std::vector<std::vector<int>> generateFullGrid() {
    std::vector<std::vector<int>> grid(GRID_SIZE, std::vector<int>(GRID_SIZE, 0));
    fillGrid(grid, 0, 0);
    return grid;
}

// Remove numbers to create a puzzle with medium difficulty (~130 empty cells)
void removeNumbers(std::vector<std::vector<int>>& grid, int emptyCells) {
    for (int i = 0; i < emptyCells; i++) {
        int row = rand() % GRID_SIZE;
        int col = rand() % GRID_SIZE;

        while (grid[row][col] == 0) { // Ensure we don't remove the same cell twice
            row = rand() % GRID_SIZE;
            col = rand() % GRID_SIZE;
        }
        grid[row][col] = 0;
    }
}

// Write the Sudoku grid to a file
void writeToFile(const std::vector<std::vector<int>>& grid, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return;
    }
    outfile << GRID_SIZE << "\n";
    for (const auto& row : grid) {
        for (int cell : row) {
            if (cell == 0) {
                outfile << "0 "; // Write 0 for empty cells
            } else {
                outfile << cell << " ";
            }
        }
        outfile << "\n";
    }

    std::cout << "Sudoku written to file: " << filename << std::endl;
    outfile.close();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <output_filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    srand(time(0)); // Ensure randomness across program runs

    // Step 1: Generate a fully solved 16x16 Sudoku grid
    auto grid = generateFullGrid();

    // Step 2: Remove numbers to create a medium difficulty puzzle
    int emptyCells = 130; // Approximate number of empty cells for medium difficulty
    removeNumbers(grid, emptyCells);

    // Step 3: Write the generated Sudoku to the specified file
    writeToFile(grid, filename);

    return 0;
}