#include <bits/stdc++.h>
using namespace std;

#define pii pair<int, int>

pii get_index(int num){
    if (num == 0)
        return {3, 1};
    for (int i = 1; i <= 9; i++){
        if (i == num)
            return {(i-1)/3, (i-1)%3};
    }
}

int main(){
    int cases, target;
    set<int> tables;
    cin >> cases;

    for(int i = 0; i < 2; i++){
        pii first = {0, 0};
        for (int j = 0; j < 10; j++){
            pii second = get_index(j);
            if (i == 0 && j == 0)
                second = {0, 0};
            if (second.first < first.first || second.second < first.second){
                continue;
            }
            for (int k = 0; k < 10; k++){
                pii third = get_index(k);
                if (third.first < second.first || third.second < second.second){
                    continue;
                }
                tables.insert(100*i+10*j+k);
            }
        }
    }
    tables.insert(200);
    
    while (cases--){
        cin >> target;
        auto second = tables.lower_bound(target);
        int second_num = *second;
        auto first = --second;
        
        if ((second_num - target) < (target - *first)){
            cout << second_num << "\n";
        }
        else{
            cout << *first << "\n";
        }
        
    }
    return 0;
}