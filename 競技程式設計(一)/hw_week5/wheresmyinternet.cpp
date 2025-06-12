
#include <bits/stdc++.h>
using namespace std;

int find(int index, int table[]){
    if (table[index] == index)
        return index;
    int prev = find(table[index], table);
    table[index] = prev;
    return prev;
}

void union1(int index1, int index2, int table[]){
    int root1 = find(index1, table);
    int root2 = find(index2, table);
    if (root1 < root2)
        table[root2] = root1;
    else
        table[root1] = root2;
}

int main()
{
    int n_houses, n_edges;
    cin >> n_houses >> n_edges;
    
    int table[n_houses+1];
    

    for (int i = 0; i < n_houses + 1; i++){
        table[i] = i;
    }
    
    int from, to;
    while (n_edges--){
        cin >> from >> to;
        
        if (from > to)
            swap(from, to);
            
        union1(from, to, table);
    }
    
    bool fully_connected = true;
    
    for (int i = 2; i < n_houses+1; i++){
        if (find(i, table) != 1){
            cout << i << '\n';
            fully_connected = false;
        }
    }
    
    if (fully_connected)
        cout << "Connected" << '\n';
    
    
    return 0;
}