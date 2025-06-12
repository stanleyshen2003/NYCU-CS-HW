#include <iostream>
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    int buildings;
    cin >> buildings;
    
    ll leftmax[buildings] = {0}, rightmax[buildings] = {0};
    
    ll building[buildings];
    
    for(int i = 0; i < buildings; i++)
        cin >> building[i];
    
    leftmax[0] = building[0];    
    for (int i = 1; i < buildings; i++)
        leftmax[i] = max(leftmax[i-1], building[i]);
        
    rightmax[buildings-1] = building[buildings-1];
    for (int i = buildings-2; i >= 0; i--)
        rightmax[i] = max(rightmax[i+1], building[i]);
        
    int ans = 0;
    
    for(int i = 0; i < buildings; i++){
        if ((min(rightmax[i], leftmax[i]) - building[i]) > ans){
            ans = min(rightmax[i], leftmax[i]) - building[i];
        }
    }
    
    cout << ans << "\n";
    
    return 0;
}