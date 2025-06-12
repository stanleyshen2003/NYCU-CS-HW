#include <bits/stdc++.h>
using namespace std;

#define tuiii tuple<int, int, int>

int find(int index, int table[]){
    if (index == table[index])
        return index;
    table[index] = find(table[index], table);
    return table[index];
}


int main(){
    int M, N;
    int from, to, weight;
    int found, sum;
    
    while (cin >> M >> N){
        if (M == 0 && N == 0)
            break;
        
        found = 0;
        sum = 0;
        set<tuiii> edges;
        set<pair<int, int>> ans;
        
        int table[M];
        for (int i = 0; i < M; i++)
            table[i] = i;
        
        while (N--){
            cin >> from >> to >> weight;
            if (from > to)
                swap(from, to);
            edges.insert({weight, from, to});
        }
        
        for (auto edge: edges){
            if (found == M-1)
                break;

            int weight = get<0>(edge);
            int x = get<1>(edge);
            int y = get<2>(edge);
            
            int root1 = find(x, table);
            int root2 = find(y, table);
            
            if (root1 == root2)
                continue;
            
            table[root1] = root2;
            found++;
            sum += weight;
            ans.insert({x, y});
        }
        
        if (found == M-1){
            cout << sum << "\n";
            for (auto edge: ans){
                cout << edge.first << " " << edge.second << "\n";
            }
        }
        else{
            cout << "Impossible\n";
        }
        
        
    }

    return 0;
}