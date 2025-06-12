#include <iostream>
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    ll left, right;
    
    cin >> left >> right;
    
    vector<pair<ll, ll>> answer;
    for (ll i = left; i <= right; i++){
        ll rightj = right / i;
        for (int j = i; j <= rightj; j++){
            ll mul = i * j;
            
            int count[10] = {0};
            ll tempi = i, tempj = j;
            while (tempi){
                count[tempi%10]++;
                tempi /= 10;
            }
            while (tempj){
                count[tempj%10]++;
                tempj /= 10;
            }
            while (mul){
                count[mul%10]--;
                mul /= 10;
            }
            
            bool correct = true;
            
            for (int k = 0; k < 10; k++){
                if (count[k]){
                    correct = false;
                    break;
                }
            }
            
            if (correct){
                answer.push_back({i, j});
            }
        }
    }
    
    printf("%d digit-preserving pair(s)\n", answer.size());
    for (int i = 0; i < answer.size(); i++){
        printf("x = %d, y = %d, xy = %d\n", answer[i].first, answer[i].second, answer[i].first * answer[i].second);
    }
    
    return 0;
}