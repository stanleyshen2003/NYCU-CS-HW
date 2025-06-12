#include <bits/stdc++.h>
using namespace std;
#define ll long long

int lnzb(int x){
    return x & (-1 * x);
}

void update(vector<vector<ll>> &table, ll index, vector<ll> gem_update, ll N){
    while (index <= N){
        for (int i = 0; i < 6; i++)
            table[index][i] += gem_update[i];
        index += lnzb(index);
    }
}

ll query(vector<vector<ll>> &table, vector<ll>& prices, ll l, ll r){
    vector<ll> sum_l(6, 0), sum_r(6, 0);
    
    while (l > 0){
        for (int i = 0; i < 6; i++)
            sum_l[i] += table[l][i];
        l -= lnzb(l);
    }
    while (r > 0){
        for (int i = 0; i < 6; i++)
            sum_r[i] += table[r][i];
        r -= lnzb(r);
    }
    
    for (int i = 0; i < 6; i++){
        sum_r[i] -= sum_l[i];
    }
    ll sum = 0;
    
    for (int i = 0; i < 6; i++){
        sum += sum_r[i] * prices[i];
        
    }
    return sum;
}

int main(){
    ll N, Q, l, r;
    
    char gem;
    cin >> N >> Q;
    
    vector<ll> prices(6, 0);
    
    vector<vector<ll>> table(N+2);
    vector<ll> gem_table(N+2, 0);
    for (int i = 0; i < N+2; i++){
        vector<ll> temp(6, 0);
        table[i] = temp;
    }
    
    for (int i = 0; i < 6; i++)
        cin >> prices[i];
        
    for (int i = 0; i < N; i++){
        cin >> gem;
        vector<ll> temp(6, 0);
        temp[gem-'1']++;
        update(table, i+1, temp, N);
        gem_table[i+1] = gem-'1';
    }
    
    ll oper, a, b;
    while (Q--){
        cin >> oper >> a >> b;
        if (oper == 1){
            vector<ll> temp(6, 0);
            temp[gem_table[a]] = -1;
            temp[b-1] = 1;
            update(table, a, temp, N);
            gem_table[a] = b-1;
        }
        else if (oper == 2){
            prices[a-1] = b;
        }
        else{
            cout << query(table, prices, a-1, b) << "\n";
        }
    }
    return 0;
}