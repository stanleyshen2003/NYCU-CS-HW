#!/bin/bash

# Compile the C code and run the executable, redirecting output to log.txt
gcc test_hw1-3.c -o test_hw1-3 && (./test_hw1-3 > log.txt &)

# Sleep for 0.5 seconds to ensure that all processes are forked and logged
sleep 0.5 && rm test_hw1-3

# Read the log file
log=$(cat log.txt && rm log.txt)

# Display the log content with newline characters preserved
echo "$log"

# Extract process IDs using awk
pids=$(echo "$log" | awk '
  /Main Process ID:/ { print $4 }
  /I'\''m the child/ { gsub(",", "", $6); print $6 }
')

# Convert the list of pids to an array
pid_array=($pids)

# Testcase answer
# |   | a | b | c | d | e | f | -> parent
# | a | - | x | x | x | x | x |
# | b | v | - | x | x | x | x |
# | c | v | v | - | x | x | x |
# | d | v | x | x | - | x | x |
# | e | v | x | x | v | - | x |
# | f | v | x | x | x | x | - |
#   -> child

answer_array=(
  "- x x x x x" 
  "v - x x x x" 
  "v v - x x x"
  "v x x - x x"
  "v x x v - x"
  "v x x x x -"
)

# Number of rows and columns
rows=6
cols=6

# Test all combinations
total_count=0
pass_count=0
for ((i=0; i<$rows; i++)); do
  answer_row="${answer_array[$i]}"
  answer_row=($answer_row)
  for ((j=0; j<$cols; j++)); do
    if [ $i -ne $j ]; then
      ((total_count++)) 
      command="./hw1-3.sh --parent ${pid_array[$j]} --child ${pid_array[$i]}"
      response=`$command`
      answer=${answer_row[$j]}
      if { [[ $response == "Yes" ]] && [[ $answer == "v" ]]; } || { [[ $response == "No" ]] && [[ $answer == "x" ]]; }; then
        echo "Testcase $total_count PASS"
        ((pass_count++))
      else
        echo "Testcase $total_count FAIL"
        echo "command: $command"
        echo "response: $response"
        echo 
      fi
    fi
  done
done

# Test whether all testcases pass
if [ $total_count -eq $pass_count ]; then
  echo "
    ################################
    #                              #
    #   You pass all testcases     #
    #                              #
    ################################
  "
else
  echo "You pass $pass_count/$total_count testcases"
fi
