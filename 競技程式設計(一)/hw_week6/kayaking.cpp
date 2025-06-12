#include <bits/stdc++.h>
using namespace std;
#define ll long long

bool check(vector<int> &kayaks, int numbers[], int strengths[], map<int, vector<pair<int, int>>> &strength_pairs, int ans){
    int strength;
    int people[3];
    for (int i = 0; i < 3; i++)
        people[i] = numbers[i];
        
    for (auto kayak : kayaks){
        int good = false;
        for (auto pairs: strength_pairs){
            if (good)
                break;
            strength = pairs.first;
            if (strength * kayak < ans){
                continue;
            }
            vector<pair<int, int>> sp = pairs.second;
            for (auto pii: sp){
                if ((pii.first == pii.second && people[pii.first] > 1)||(pii.first != pii.second && people[pii.first] > 0 && people[pii.second] > 0)){
                    people[pii.first]--;
                    people[pii.second]--;
                    good = true;
                    break;
                }
            }
        }

        if (!good){
            return false;
        }
    }
    return true;
}

int main(){
    int numbers[3], strengths[3], total = 0, k_constant;
    for (int i = 0; i < 3; i++){
        cin >> numbers[i];
        total += numbers[i];
    }
    for (int i = 0; i < 3; i++)
        cin >> strengths[i];
    
    map<int, vector<pair<int, int>>> strength_pairs;
    for (int i = 0; i < 3; i++){
        for (int j = i; j < 3; j++){
            strength_pairs[strengths[i] + strengths[j]].push_back({i, j});
        }
    }
    
    total /= 2;
    vector<int> kayaks;
    
    for (int i = 0; i < total; i++){
        cin >> k_constant;
        kayaks.push_back(k_constant);
    }
    
    sort(kayaks.begin(), kayaks.end());
    
    int left = 2 * strengths[0] * kayaks[0];
    int right = 2 * strengths[2] * kayaks[kayaks.size()-1];
    
    while (left < right - 1){
        int temp = left + (right - left) / 2;
        if (check(kayaks, numbers, strengths, strength_pairs, temp)){
            left = temp;
        }
        else {
            right = temp - 1;
        }
        
    }
    
    if (check(kayaks, numbers, strengths, strength_pairs, right))
        cout << right << endl;
    else
        cout << left << endl;
        
    return 0;
}