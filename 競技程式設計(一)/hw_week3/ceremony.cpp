#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, height, temp;
    cin >> n;
    
    int ans = n;
    vector<int> buildings;
    
    for (int i = 0; i < n; i++){
        cin >> height;
        buildings.push_back(height);
    }
    
    sort(buildings.begin(), buildings.end());
    
    for (int i = 0; i < n; i++){
        temp = buildings[i] + n - i - 1;
        if (temp < ans)
            ans = temp;
    }
    
    cout << ans << "\n";

    return 0;
}