# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -g

# Target executable name
TARGET = generate_sudoku

# Source files
SRC = generate-sudoku-16x16.cpp 

# Object files (automatic substitution for .o files)
OBJ = $(SRC:.cpp=.o)

# Build target
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJ)
