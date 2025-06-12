#include <bits/stdc++.h>
using namespace std;
#define pii pair<int, int>

int tsp(int mask, int curr, vector<vector<int>>& memo, vector<vector<int>> &cost){
    if (mask == (1<<cost.size())-1)
        return cost[curr][0];
        
    if (memo[mask][curr] != -1)
        return memo[mask][curr];
    
    int ans = INT_MAX;
    
    for (int i = 0; i < cost.size(); i++){
        if (mask&(1<<i))
            continue;
        ans = min(ans, cost[curr][i] + tsp(mask|(1<<i), i, memo, cost));
    }

    memo[mask][curr] = ans;
    return ans;
}

int main() {
    int n, x, y;
    string name;
    cin >> n;
    
    unordered_map<string, vector<pii>> pokemons;
    
    for (int i = 0; i < n; i++){
        cin >> x >> y >> name;
        pokemons[name].push_back({x, y});
    }
    
    vector<vector<int>> indeces(1, vector<int>());
    vector<vector<int>> new_indeces;
    
    for (auto pokemon: pokemons){
        for (int i = 0; i < pokemon.second.size(); i++){
            for (auto index: indeces){
                vector<int> temp = index;
                temp.push_back(i);
                new_indeces.push_back(temp);
            }
        }
        indeces = new_indeces;
        new_indeces.clear();
    }
    int ans = INT_MAX;
    for (auto index: indeces){
        
        vector<pii> links;
        links.push_back({0, 0});
        for (auto pokemon: pokemons){
            links.push_back(pokemon.second[index[links.size()-1]]);
        }
        n = links.size();
        vector<vector<int>> memo((1<<(n)), vector<int>(n, -1));
        vector<vector<int>> costs(n, vector<int>(n));

        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                costs[i][j] = abs(links[i].first - links[j].first) + abs(links[i].second - links[j].second);
            }
        }

        int newans = tsp(1, 0, memo, costs);

        if (ans > newans)
            ans = newans;
    }
    cout << ans << "\n";
    
    return 0;
}