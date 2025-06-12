#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    string original;
    
    int top = -1;
    char stack[1000000];
    
    cin >> original;

    for (char c:original){
        if (c != '<')
            stack[++top] = c;
        else
            top--;
    }
        
    for (int i = 0; i < top+1; i++)
        cout << stack[i];

    cout << "\n";

    return 0;
}