#!/bin/bash

path="0"
# Read parent pid and child pid
while [ "$#" -gt 0 ]; do
  case "$1" in
    --parent)
      parent="$2"
      shift 2
      ;;
    --child)
      child="$2"
      shift 2
      ;;
    --path)
      path="1"
      shift 1
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# Check if parent or child is empty
if [ -z "$parent" ] || [ -z "$child" ]; then
  echo "Usage: $0 --parent PARENT_PID --child CHILD_PID"
  exit 1
fi

# Traverse the process hierarchy to find the grandparent
ancestorfound="No"
current_pid="$child"
track="$child" 
while [ "$current_pid" != 1 ]; do
  current_ppid=$(ps -o ppid= -p "$current_pid" | tr -d ' ')
  track="${current_ppid} -> ${track}"
  if [ "$current_ppid" = "$parent" ]; then
    ancestorfound="Yes"
    break
  fi

  current_pid="$current_ppid"
done

echo "$ancestorfound"
if [ "$ancestorfound" = "Yes" ]; then
  if [ "$path" = "1" ]; then
    echo "$track"
  fi
fi