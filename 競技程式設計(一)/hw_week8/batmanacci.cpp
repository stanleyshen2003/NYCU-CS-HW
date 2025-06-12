#include <vector>
#include <iostream>
using namespace std;

#define ll unsigned long long

int main(){
    ll N, K;
    cin >> N >> K;
    ll length[100];
    length[0] = length[1] = length[2] = 1;
    for(int i = 3; i < 100; i++){
        length[i] = length[i-1] + length[i-2];
    }
    
    int max_less_than_K = 3;
    while(length[max_less_than_K]<K){
        max_less_than_K++;
    }
    
    if((N - max_less_than_K) %2){
        N = max_less_than_K+1;
    }
    else{
        N = max_less_than_K;
    }
        
    while(1){
        if(N==1){
            cout << 'N';
            break;
        }
        if(N==2){
            cout << 'A';
            break;
        }

        if(K <= length[N-2]){
            N-=2;
        }
        else{
            K-=length[N-2];
            N-=1;
        }
        
    }
    return 0;
}
