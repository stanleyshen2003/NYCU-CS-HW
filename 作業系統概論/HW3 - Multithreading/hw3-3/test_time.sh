#!/bin/bash

echo "hw3-2.cpp"
g++ hw3-3_old.cpp -o thread_old.out -lpthread
g++ hw3-3_serial.cpp -o serial.out
g++ ../hw3-3.cpp -o thread.out -lpthread

echo "serial:"
echo `time ./serial.out < testcase/case3.txt`
echo "old:"
echo `time ./thread_old.out -t 4 < testcase/case3.txt`
echo "new:"
echo `time ./thread.out -t 4 < testcase/case3.txt`