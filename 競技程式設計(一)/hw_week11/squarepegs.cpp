# include <bits/stdc++.h>
using namespace std;
int main()
{
    int K, M, N;
    cin >> K >> M >> N;
    int ans = 0;
    
    vector<int> plots(K), houses(M+N);
    map<int, int> mapping;
    for (int i = 0; i < K; i++)
        cin >> plots[i];
        
    for (int i = 0; i < M; i++)
        cin >> houses[i];
        
    int side;
    for (int i = 0; i < N; i++){
        cin >> side;
        side = side / sqrt((double)2);
        houses[i+M] = side;
    }
    
    sort(plots.begin(), plots.end());
    for (int house: houses)
        mapping[house]++;
        
    for (int plot: plots){
        auto it = mapping.lower_bound(plot);
        if (it == mapping.begin())
            continue;
        ans++;
        it--;
        (*it).second--;
        if ((*it).second == 0)
            mapping.erase(it);
    }
    
    cout << ans << "\n";

    return 0;
}