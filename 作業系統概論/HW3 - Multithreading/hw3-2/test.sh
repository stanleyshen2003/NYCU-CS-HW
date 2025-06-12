#!/bin/bash

echo "hw3-2.cpp"
g++ --std=c++2a hw3-2.cpp 
if [ $? -eq 0 ]; then
  # small testcases: test for deadlock or unconverged
  for testcase in 1 2 3
  do
    tle=0
    for ((i = 0; i < 40; i++)); do
      timeout 3.5 ./a.out < testcase/case$testcase.txt > ans.txt
      
      if [ $? -ne 0 ]; then
        tle=1
        break
      fi
    done
    if [ $tle -eq 0 ]; then
      python3 val.py $testcase ans.txt
      if [ $? -eq 0 ]; then
        echo "testcase $testcase: AC"
      else
        echo "testcase $testcase: WA"
      fi
    else
      echo "testcase $testcase: TLE"
    fi
  done

  # large testcase: test for busy wait
  for testcase in 4 5 6
  do
    timeout 3.5 ./a.out < testcase/case$testcase.txt > ans.txt
    if [ $? -eq 0 ]; then
      python3 val.py $testcase ans.txt
      if [ $? -eq 0 ]; then
        echo "testcase $testcase: AC"
      else
        echo "testcase $testcase: WA"
      fi
    else
      echo "testcase $testcase: TLE"
    fi
  done
else
  echo "CE"
fi
