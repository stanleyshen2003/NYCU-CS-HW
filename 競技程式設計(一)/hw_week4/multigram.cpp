#include <iostream>
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    string s;
    int sl;
    bool correct;
    cin >> s;
    
    for (sl = 1; sl < s.length()/2 + 1; sl++){
        if (s.length() % sl != 0)
            continue;
        
        int table[27] = {0};
        
        for (int j = 0; j < sl; j++){
            table[s[j] - 'a']++;
        }
        
        correct = true;
        
        for (int i = sl; i < s.length(); i += sl){
            int temp[27] = {0};
            for (int j = 0; j < sl; j++){
                temp[s[i+j] - 'a']++;
            }
            
            for (int k = 0; k < 27; k++){
                if (temp[k] != table[k]){
                    correct = false;
                    break;
                }
            }
            
            if (!correct)
                break;
        }
        if (correct)
            break;
    }
    

    if (correct){
        for (int i = 0; i < sl; i++){
            cout << s[i];
        }
        cout << "\n";
    }
    else
        cout << "-1\n";
    
    return 0;
}