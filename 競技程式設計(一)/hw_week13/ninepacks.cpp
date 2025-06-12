#include <bits/stdc++.h>
using namespace std;

int main(){
    int H, B;
    
    cin >> H;
    vector<int> hotdogs(H);
    for (int i = 0; i < H; i++)
        cin >> hotdogs[i];
        
    cin >> B;
    vector<int> buns(B);
    for (int i = 0; i < B; i++)
        cin >> buns[i];
    
    vector<int> hotdog_t(100001, INT_MAX);
    hotdog_t[0] = 0;
    for (int i = 0; i < H; i++){
        for (int j = 100000; j >= hotdogs[i]; j--){
            if (hotdog_t[j-hotdogs[i]] != INT_MAX){
                hotdog_t[j] = min(hotdog_t[j], hotdog_t[j-hotdogs[i]]+1);
            }
        }
    }
    
    vector<int> bun_t(100001, INT_MAX);
    bun_t[0] = 0;
    for (int i = 0; i < B; i++){
        for (int j = 100000; j >= buns[i]; j--){
            if (bun_t[j-buns[i]] != INT_MAX){
                bun_t[j] = min(bun_t[j], bun_t[j-buns[i]]+1);
            }
        }
    }
    
    int min_total = INT_MAX;
    for (int i = 1; i < 100001; i++){
        if (bun_t[i]!=INT_MAX && hotdog_t[i] != INT_MAX){
            min_total = min(min_total, bun_t[i] + hotdog_t[i]);
        }
    }
    
    if (min_total == INT_MAX)
        cout << "impossible\n";
    else
        cout << min_total << "\n";
    return 0;
}