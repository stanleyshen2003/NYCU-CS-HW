#include <bits/stdc++.h>
using namespace std;

int totalCost(int mask, int curr, int n, 
              vector<vector<bool>> &grid, vector<vector<int>> &memo) {
                
    if (mask == (1<<n) - 1) {
        if (grid[curr][0])
            return 1;
        return 0;
    }
    
    if (memo[mask][curr]!= -1)
        return memo[mask][curr];

    int ans = 0;
    
    for (int i = 0; i < n; i++){
        if (mask & (1<<i) || grid[i][curr] == false)
            continue;
        ans += totalCost(mask | (1<<i), i, n, grid, memo);
    }

    return memo[mask][curr] = ans;
}

int main() {
    int T, n, k, x, y;
    cin >> T;
    
    for (int i = 1; i <= T; i++){
        cin >> n >> k;
        vector<vector<bool>> grid(n, vector<bool>(n, true));
        vector<vector<int>> memo(1 << n, vector<int>(n, -1));

        while (k--){
            cin >> x >> y;
            x--; y--;
            grid[x][y] = grid[y][x] = false;
        }

        int total = totalCost(1, 0, n, grid, memo);
        
        cout << "Case #" << i << ": " << (total/2) % 9901 << "\n";
    }
    
    return 0;
}