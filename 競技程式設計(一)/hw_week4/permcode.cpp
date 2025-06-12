#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <unordered_map>
using namespace std;
#define ll long long


int main()
{
    string S, P, C;
    int key, d;
    while(1){
        cin >> key;
        if(key == 0)
            break;
        cin >> S >> P >> C;
        d = (int)(pow((double)C.length(), 1.5) + key) % C.length();
        unordered_map<char, int> S_inverse, P_inverse;
        for(int i = 0; i < S.length(); i++){
            S_inverse[S[i]] = i;
            P_inverse[P[i]] = i;
        }
        char ans[C.length()];
        ans[d] = P[S_inverse[C[d]]];
        //cout << "d: " << d <<" "<< ans[d] << endl;
        for(int i = (d-1 + C.length())%C.length(); i!=d; i = (i+C.length()-1) % C.length()){
            ans[i] = P[S_inverse[C[i]] ^ S_inverse[ans[(i+1)%C.length()]]];
        }
        for(int i = 0; i<C.length(); i++)
            cout << ans[i];
        cout << endl;
    }
    
    return 0;
}
