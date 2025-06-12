#include <iostream>
#include <set>
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int N, T;
    int ans = 0;
    
    cin >> N >> T;
    
    bool occupied[T];
    for (int i = 0; i < T; i++)
        occupied[i] = false;
    vector<pair<int, int>> customers;
    
    int cash, ttl;
    
    for (int i = 0; i < N; i++){
        cin >> cash >> ttl;
        customers.push_back({cash, ttl});
    }
    
    sort(customers.begin(), customers.end());
    
    for (int i = N-1; i >= 0; i--){
        if (T == 0)
            break;
            
        cash = customers[i].first;
        ttl = customers[i].second;
        
        for (int j = ttl; j >= 0; j--){
            if (!occupied[j]){
                ans += cash;
                occupied[j] = true;
                T--;
                break;
            }
        }
    }
    
    cout << ans << "\n";

    return 0;
}