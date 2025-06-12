#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    double score, target;
    
    int target_i;

    vector<int> scores;
    
    for (int i = 0; i < 4; i++){
        cin >> score;
        scores.push_back((int)((score*100)+0.5));
    }
        
    cin >> target;
    target_i = (int)((target*100)+0.5);
    target_i *= 3;
    
    sort(scores.begin(), scores.end());
    
    int minimum = scores[0] + scores[1] + scores[2];
    int maximum = scores[1] + scores[2] + scores[3];
    
    if (maximum <= target_i)
        cout << "infinite\n";
    else if (minimum > target_i)
        cout << "impossible\n";
    else {
        int ans = target_i - scores[1] - scores[2];
        bool pad = (ans % 100) < 10;
        cout << ans / 100 << '.';
        if (pad)
            cout << '0';
        cout << ans % 100;
    }
    
    return 0;
}