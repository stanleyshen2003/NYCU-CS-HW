#include <iostream>
#include <stack>
#include <bits/stdc++.h>

using namespace std;

bool valid(char x){
    if(x>='a' && x <= 'f')
        return true;
    if(x >= 'A' && x <= 'F')
        return true;
    if(x >= '0' && x <= '9')
        return true;
    return false;
}

int main()
{
    string s;
    int ans;
    string num = "";
    int start;
    
    while(cin >> s){
        start = -1;
        for(int i = 1; i < s.length(); i++){
            if(start != -1){
                //cout << start;

                if(valid(s[i]) && i-start+1 < 10 && i!=s.length()-1)
                    continue;
                else{
                    if(valid(s[i])){
                        num = s.substr(start, 10);
                    }
                    else{
                        num = s.substr(start, i-start);
                    }
                    start = -1;
                    if(num.length()==2)
                        continue;
                    cout << num << " " << stoul(num.substr(2, num.length()-2), 0, 16) << endl;
                    
                }
            }
            if(s[i-1] == '0' && (s[i] == 'x' || s[i] == 'X'))
                start = i-1;
        }
    }
    return 0;
}
