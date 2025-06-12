import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
path = 'data/threads-parallel-dfs-bfs.csv'
data = pd.read_csv(path)

# Filter data to start from Thread_Count=2
data = data[data['Thread_Count'] >= 2]

# Extract thread counts
thread_counts = data['Thread_Count']

# --- Backtracking Methods ---
# Get the baseline time for Backtracking Serial at Thread_Count=2
serial_baseline_backtracking = data.loc[data['Thread_Count'] == 2, 'backtracking-serial'].values[0]

if serial_baseline_backtracking <= 0:
    raise ValueError("backtracking-serial Time at Thread_Count=2 must be greater than 0.")

# Calculate speedups
speedup_backtracking_pthread = serial_baseline_backtracking / data['backtracking-dfs-pthread']
speedup_backtracking_openmp = serial_baseline_backtracking / data['backtracking-dfs-openmp']

# Plot Backtracking Speedup
plt.figure(figsize=(12, 6))

# Plot the Serial line as a flat line at speedup = 1
plt.plot(thread_counts, [1] * len(thread_counts), 
         marker='o', linestyle='-', color='blue', label='Backtracking-Serial')

plt.plot(thread_counts, speedup_backtracking_pthread, 
         marker='s', linestyle='-', color='green', label='Backtracking-DFS-Pthread')

plt.plot(thread_counts, speedup_backtracking_openmp, 
         marker='^', linestyle='-', color='orange', label='Backtracking-DFS-OpenMP')

plt.title('Backtracking Methods Speedup Comparison')
plt.xlabel('Thread Count')
plt.ylabel('Speedup (Relative to Backtracking-Serial)')

# Set x-axis ticks with interval = 1
x_ticks = np.arange(thread_counts.min(), thread_counts.max() + 1, 1)
plt.xticks(x_ticks)

plt.legend()
plt.grid(True)

plt.savefig('backtracking_speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Bruteforce Methods ---
# Get the baseline time for Bruteforce Serial at Thread_Count=2
serial_baseline_bruteforce = data.loc[data['Thread_Count'] == 2, 'bruteforce-serial'].values[0]

if serial_baseline_bruteforce <= 0:
    raise ValueError("bruteforce-serial Time at Thread_Count=2 must be greater than 0.")

# Calculate speedups
speedup_bruteforce_pthread = serial_baseline_bruteforce / data['bruteforce-bfs-pthread']
speedup_bruteforce_openmp = serial_baseline_bruteforce / data['bruteforce-bfs-openmp']

# Plot Bruteforce Speedup
plt.figure(figsize=(12, 6))

# Plot the Serial line as a flat line at speedup = 1
plt.plot(thread_counts, [1] * len(thread_counts), 
         marker='o', linestyle='-', color='blue', label='Bruteforce-Serial')

plt.plot(thread_counts, speedup_bruteforce_pthread, 
         marker='D', linestyle='-', color='red', label='Bruteforce-BFS-Pthread')

plt.plot(thread_counts, speedup_bruteforce_openmp, 
         marker='*', linestyle='-', color='purple', label='Bruteforce-BFS-OpenMP')

plt.title('Bruteforce Methods Speedup Comparison')
plt.xlabel('Thread Count')
plt.ylabel('Speedup (Relative to Bruteforce-Serial)')

# Set x-axis ticks with interval = 1
plt.xticks(x_ticks)

plt.legend()
plt.grid(True)

plt.savefig('bruteforce_speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
