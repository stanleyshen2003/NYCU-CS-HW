#include <iostream>
using namespace std;

bool found(int * cache,int n, int address){
    for(int i=0;i<n;i++){
        if(cache[i] == address)
            return true;
    }
    return false;
}


int main(){
    int n;
    int stream_size;
    int miss_count = 0;
    cin >> n >> stream_size;
    int *cache = new int[n];
    for(int i=0;i<n;i++){
        cache[i] = -9999;
    }
    int address;
    int block;
    for(int i = 0;i < stream_size; i++){
        cin >> address;

        if (found(cache, n, address))
            continue;
        
        miss_count ++;
        cache[i % n] = address;
        
    }
    cout << "Total Cache Misses:" << miss_count;
    delete[] cache;
    return 0;
}

