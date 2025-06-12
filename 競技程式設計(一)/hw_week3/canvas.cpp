#include <iostream>
#include <set>
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int testcase;
    int n_canva;
    int canva_size;
    
    cin >> testcase;
    
    while (testcase--){
        cin >> n_canva;
        priority_queue<ll, vector<ll>, greater<ll>> canvas;
        ll ans = 0;
        
        while (n_canva--){
            cin >> canva_size;
            canvas.push(canva_size);
        }
        
        while (canvas.size() != 1){
            ll first = canvas.top();
            canvas.pop();
            ll second = canvas.top();
            canvas.pop();
            ans += first + second;
            canvas.push(first+second);
        }
        cout << ans << "\n";
    }

    return 0;
}