#include <bits/stdc++.h>
using namespace std;


int find(int index, int table[]){
    if (index == table[index])
        return index;
    table[index] = find(table[index], table);
    return table[index];
}


int main(){
    int size;
    cin >> size;
    
    int grid[size][size];
    
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            cin >> grid[i][j];
    
    set<pair<int, pair<int, int>>> edges;
    
    for (int i = 0; i < size; i++)
        for (int j = i+1; j < size; j++)
            edges.insert(make_pair(grid[i][j], make_pair(i, j)));
            
    int table[size];
    for (int i = 0; i < size; i++){
        table[i] = i;
    }
    
    int amount = size-1;
    set<pair<int, int>> ans;
    
    for (auto edge: edges){
        if (!amount)
            break;
        int edge1 = edge.second.first;
        int edge2 = edge.second.second;
        int root1 = find(edge1, table);
        int root2 = find(edge2, table);
        if (root1 == root2)
            continue;
        table[root1] = root2;
        ans.insert({edge1, edge2});
        amount--;
    }
    
    for (auto edge: ans){
        cout << edge.first+1 << " " << edge.second+1 << "\n";
    }

    return 0;
}