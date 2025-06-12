
#include <bits/stdc++.h>
using namespace std;

int directions[4][2] = {{1,0}, {-1, 0}, {0, 1}, {0, -1}};

void set_connected(int i, int j, int value, vector<vector<int>>& garden, vector<vector<bool>>& hold_water){
    if (i < 0 || j < 0 || i >= garden.size() || j >= garden[1].size())
        return;
        
    if (garden[i][j] != value)
        return;
    
    if (hold_water[i][j] == false)
        return;
        
    hold_water[i][j] = false;
        
    for (int k = 0; k < 4; k++){
        set_connected(i + directions[k][0], j + directions[k][1], value, garden, hold_water);
    }
    
}


int main()
{
    int m, n;
    cin >> n >> m;
    
    vector<vector<int>> garden(m, vector<int>(n, 0));
    vector<vector<bool>> hold_water(m, vector<bool>(n, true));
    
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            cin >> garden[i][j];
            
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (i != 0 && garden[i-1][j] < garden[i][j]){
                hold_water[i][j] = false;
                continue;
            }
            if (i != m-1 && garden[i+1][j] < garden[i][j]){
                hold_water[i][j] = false;
                continue;
            }
            if (j != 0 && garden[i][j-1] < garden[i][j]){
                hold_water[i][j] = false;
                continue;
            }
            if (j != n-1 && garden[i][j+1] < garden[i][j]){
                hold_water[i][j] = false;
                continue;
            }
        }
    }
    
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (hold_water[i][j] == false){
                hold_water[i][j] = true;
                set_connected(i, j, garden[i][j], garden, hold_water);
            }
        }
    }
    
    int ans = 0;
    
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (hold_water[i][j])
                ans++;
        }
    }
    
    cout << ans << '\n';
    
    return 0;
}