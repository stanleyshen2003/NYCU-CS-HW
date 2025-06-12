#include <bits/stdc++.h>
using namespace std;
#define ll long long


ll search(vector<ll> &sequence, ll left, ll right, ll minimum, ll maximum){
    if (right < left)
        return 0;
    
    ll mid = left + (right - left) / 2;
    
    ll found = 0;
    if (sequence[mid] < maximum && sequence[mid] > minimum)
        found = 1;

    return found + search(sequence, left, mid-1, minimum, min(maximum, sequence[mid])) + search(sequence, mid+1, right, max(minimum, sequence[mid]), maximum);
}

int main(){
    ll n, m, a, c, x0;
    
    cin >> n >> m >> a >> c >> x0;
    
    vector<ll> sequence;
    
    while (n--){
        x0 = (a * x0 + c) % m;
        sequence.push_back(x0);
    }
    
    cout << search(sequence, 0, sequence.size()-1, LLONG_MIN, LLONG_MAX) << "\n";

    return 0;
}