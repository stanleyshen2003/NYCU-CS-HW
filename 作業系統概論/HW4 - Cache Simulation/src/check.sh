#!/usr/bin/bash

if [ ! $# -eq 5 ]; then
    echo "Usage: ./check.sh <part1_executable> <part2_LRU_executable> <part2_LFU_executable> <testcaseDir> <ansDir>"
    exit 1
fi
part1_executable=$1
part2_LRU_executable=$2
part2_LFU_executable=$3
testcaseDir=$4
ansDir=$5

tmpfile="${ansDir}/tmp.txt"
touch $tmpfile

if [ ! -d "${ansDir}" ]; then
    echo "Error: ${ansDir} does not exist"
    exit 1
fi
if [ ! -d "${testcaseDir}" ]; then
    echo "Error: ${testcaseDir} does not exist"
    exit 1
fi

failed=0
for i in {0..999}
do
    filename="${testcaseDir}/part1/tc_${i}.txt"
    ./$part1_executable < $filename > $tmpfile
    diff $tmpfile "${ansDir}/part1/ans_${i}.txt" > /dev/null
    if [ $? -eq 0 ]; then
        # color it green
        echo -e "\033[32m part1 testcase ${i} passed \033[0m"
    else
        # color it red
        echo -e "\033[31m part1 testcase ${i} failed \033[0m"
        failed=1
    fi
done    
echo "part1 testcases finished, passed: $((1000-failed)), failed: $failed"

failed=0
for i in {0..999}
do
    filename="${testcaseDir}/part2/tc_${i}.txt"
    ./$part2_LRU_executable < $filename > $tmpfile
    diff $tmpfile "${ansDir}/part2_LRU/ans_${i}.txt" > /dev/null
    if [ $? -eq 0 ]; then
        # color it green
        echo -e "\033[32m part2_LRU testcase ${i} passed \033[0m"
    else
        # color it red
        echo -e "\033[31m part2_LRU testcase ${i} failed \033[0m"
        failed=1
    fi
done
echo "part2_LRU testcases finished, passed: $((1000-failed)), failed: $failed"

failed=0
for i in {0..999}
do
    filename="${testcaseDir}/part2/tc_${i}.txt"
    ./$part2_LFU_executable < $filename > $tmpfile
    diff $tmpfile "${ansDir}/part2_LFU/ans_${i}.txt" > /dev/null
    if [ $? -eq 0 ]; then
        # color it green
        echo -e "\033[32m part2_LFU testcase ${i} passed \033[0m"
    else
        # color it red
        echo -e "\033[31m part2_LFU testcase ${i} failed \033[0m"
        failed=1
    fi
done
echo "part2_LFU testcases finished, passed: $((1000-failed)), failed: $failed"