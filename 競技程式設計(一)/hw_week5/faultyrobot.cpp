#include <bits/stdc++.h>
using namespace std;

int main(){
    int nodes, edges;
    cin >> nodes >> edges;
    
    vector<int> force_edges(nodes+1);
    vector<vector<int>> normal_edges(nodes+1);
    
    for(int i = 1; i < nodes+1; i++)
        force_edges[i] = -1;
        
    int from, to;
    while (edges--){
        cin >> from >> to;
        if (from < 0)
            force_edges[-1*from] = to;
        else
            normal_edges[from].push_back(to);
    }
    
    unordered_set<int> path;
    path.insert(1);
    int current_node = 1;
    while (force_edges[current_node] != -1){
        current_node = force_edges[current_node];
        if (path.find(current_node)!= path.end())
            break;
        path.insert(current_node);
    }
    
    unordered_set<int> new_path;
    new_path = path;
    
    for (auto node:path){
        for (auto possible_path:normal_edges[node]){
            new_path.insert(possible_path);
            current_node = possible_path;
            while (force_edges[current_node] != -1){
                current_node = force_edges[current_node];
                if (new_path.find(current_node)!= new_path.end())
                    break;
                new_path.insert(current_node);
            }
        }
    }
    
    int ans = 0;
    
    for (auto node: new_path){
        if (force_edges[node] == -1)
            ans++;
    }

    cout << ans << "\n";

    return 0;
}