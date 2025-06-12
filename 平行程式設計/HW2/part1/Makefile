# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O2 -pthread -lm

# Target executable
TARGET = pi.out

# Source files
SRCS = pi.c

# Default rule to build the program
all: $(TARGET)

# Rule to link and compile the program
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET)
