#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    ll T, N, ans;
    int target;
    cin >> T;
    
    while (T--){
        cin >> N;
        
        ans = 0;
        ll table[N];
        unordered_map<ll, ll> record;
        
        for (int i = 0; i < N; i++)
            cin >> table[i];
            
        for (int i = 1; i < N; i++)
            table[i] += table[i-1];
            
        for (int i = 0; i < N; i++){
            if (table[i] == 47)
                ans ++;
            record[(-1) * table[i]]++;
            ans += record[47 - table[i]];
        }
        
        cout << ans << "\n";
    }

    return 0;
}