#include <bits/stdc++.h>
using namespace std;

#define ll long long

int main(){
    int M, N;
    cin >> M >> N;
    
    int directions[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    
    ll table[M][N];
    priority_queue<tuple<ll, int, int>, vector<tuple<ll, int, int>>, greater<tuple<ll, int, int>>> pq;
    
    for(int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            cin >> table[i][j];
            
    for (int i = 0; i < M; i++)
        pq.push({table[i][0], i, 0});
        
    int addx, addy, nextx, nexty;
    ll ans[M][N];
    
    for(int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            ans[i][j] = -1;
    
    while (1){
        if (addy == N-1){
            cout << ans[addx][addy] << "\n";
            break;
        }
        
        tuple<ll, int, int> add = pq.top();
        addx = get<1>(add);
        addy = get<2>(add);
        pq.pop();
        
        if (ans[addx][addy] != -1)
            continue;
            
        ans[addx][addy] = get<0>(add);
        
        for (int i = 0; i < 4; i++){
            nextx = addx + directions[i][0];
            nexty = addy + directions[i][1];
            
            if (nextx >= 0 && nextx < M && nexty >= 0 && nexty < N && ans[nextx][nexty] == -1)
                pq.push({max(table[nextx][nexty], get<0>(add)), nextx, nexty});
        }
    }

    return 0;
}