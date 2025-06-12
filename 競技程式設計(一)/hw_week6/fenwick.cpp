#include <bits/stdc++.h>
using namespace std;
#define ll long long

ll lnzb(ll a){
    return a & (-1 * a);
}

void add(ll tree[], ll a, ll b, ll N){
    while (a <= N){
        tree[a] += b;
        a += lnzb(a);
    }
}

ll query(ll tree[], ll a){
    ll ans = 0;
    while (a > 0){
        ans += tree[a];
        a -= lnzb(a);
    }
    return ans;
}

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    ll N, Q;
    cin >> N >> Q;
    char operation;
    ll a, b;
    
    ll fenwicktree[N+1] = {0};
    for (int i = 0; i < N+1; i++){
        fenwicktree[i] = 0;
    }
    
    while (Q--){
        cin >> operation;
        
        if (operation == '+'){
            cin >> a >> b;
            a += 1;
            add(fenwicktree, a, b, N+1);
        }
        else{
            cin >> a;
            ll ans = query(fenwicktree, a);
            cout << ans << "\n";
            
        }

    }
    
    
    

    return 0;
}