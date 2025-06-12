#include <iostream>
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main(){
    int number;
    cin >> number;
    number++;
    double temp = 1;
    double ans = 1;
    for (int i = 1; i < number; i++){
        temp /= i;
        ans += temp;
    }
    
    printf("%.20f\n", ans);
    
    return 0;
}