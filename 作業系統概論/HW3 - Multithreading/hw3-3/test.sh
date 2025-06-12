#!/bin/bash

echo "hw3-2.cpp"
g++ ../hw3-3.cpp -o thread.out -lpthread
g++ hw3-3_serial.cpp -o serial.out

correct_count=0
incorrect_count=0

# Function to compare outputs
compare_outputs() {
    input=$1
    output_thread=$(./thread.out -t 4 <<< "$input")
    output_serial=$(./serial.out <<< "$input")

    if [ "$output_thread" != "$output_serial" ]; then
        #echo "Outputs are different for input: $input"
        ((incorrect_count++))
    else
        #echo "Outputs are the same for input: $input"
        ((correct_count++))
    fi
}

# Check if arguments are provided or ask for user input
if [ "$#" -eq 2 ]; then
    min=$1
    max=$2

    for ((i=min; i<=max; i++)); do
        compare_outputs $i
    done
else
    echo "Usage: ./test.sh [min] [max]"
    echo "Or run without arguments to enter a single integer input."
    exit 1
fi

# Display the count of correct and incorrect outputs
echo "Correct outputs count: $correct_count"
echo "Incorrect outputs count: $incorrect_count"