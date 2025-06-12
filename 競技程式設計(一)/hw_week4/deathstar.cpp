#include <iostream>
#include <bits/stdc++.h>
using namespace std;
int main(){
    int size;
    cin >> size;
    unsigned int table[size][size];
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cin >> table[i][j];
        }
    }
    
    int ans[size] = {0};
    
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if (i != j){
                ans[i] = ans[i] | table[i][j];
            }
        }
    }
    
    for (int i = 0; i < size-1; i++){
        cout << ans[i] << " ";
    }
    
    cout << ans[size-1] << "\n";
    
    return 0;
}