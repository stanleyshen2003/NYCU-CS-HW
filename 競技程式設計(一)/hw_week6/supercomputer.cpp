#include <bits/stdc++.h>
using namespace std;

int lnzb(int x){
    return x & (-1 * x);
}

void flip(int table[], int index, int update, int N){
    while (index <= N){
        table[index] += update;
        index += lnzb(index);
    }
}

int query(int table[], int l, int r){
    int l_sum = 0, r_sum = 0;
    
    while (l > 0){
        l_sum += table[l];
        l -= lnzb(l);
    }
    while (r > 0){
        r_sum += table[r];
        r -= lnzb(r);
    }
    return r_sum - l_sum;
}

int main(){
    int N, K, l, r;
    char operation;
    cin >> N >> K;
    
    int table[N+1] = {0};
    int flip_table[N+1] = {0};
    
    while (K--){
        cin >> operation;
        
        if (operation == 'F'){
            cin >> l;
            // l++;
            if (flip_table[l]){
                flip(table, l, -1, N);
                flip_table[l] = 0;
            }
            else {
                flip(table, l, 1, N);
                flip_table[l] = 1;
            }
        }
        else {
            cin >> l >> r;
            l--;
            cout << query(table, l, r) << "\n";
        }

    }

    return 0;
}