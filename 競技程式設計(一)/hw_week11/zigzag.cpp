# include <bits/stdc++.h>
using namespace std;
#define pii pair<int, int>

int main(){
    int num, length = 2;
    cin >> num;
    
    length += (num-1)/25;
    string s = "a";
    
    if (num <= 25){
        cout << 'a' << (char)('a' + num) << "\n";
        return 0;
    }
    
    if(num % 25 == 0) {
        while(num >= 25) {
            num -= 25;
            if(s[s.size()-1] == 'a') {
                s.push_back('z');
            }
            else {
                s.push_back('a');
            }
        }
        cout << s << endl;
        return 0;
    }
    int rem = num % 25;
    bool secondrem = rem%2==1;

    s += 'n' + rem/2;

    while(num >= 25) {
        num -= 25;
        if(s[s.size()-1] == 'a') {
            s.push_back('z');
        }
        else {
            s.push_back('a');
        }
    }

    if(!secondrem) {
        if(s[s.size()-1] == 'a') {
            s[s.size()-1]++;
        }
        else {
            s[s.size()-1]--;
        }
    }
    
    cout << s << endl;
    
    return 0;
}