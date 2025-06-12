#include <iostream>
#include <vector>
using namespace std;

int k,n,stream_size, n_set;
struct addr{
    int value;
    int use_count;
};

bool found(struct addr ** cache,int address, int time){
    for(int i=0;i<n_set;i++){
        for(int j = 0; j < k; j++){
            if (cache[i][j].value == address){
                cache[i][j].use_count++;
                return true;
            }
        }
    }
    return false;
}

void put(addr** cache, int put_addr, int value){
    int set = put_addr % n_set;
    // if empty space
    for(int i = 0; i < k; i++){
        if (cache[set][i].value == -1){
            cache[set][i].value = value;
            cache[set][i].use_count = 0;
            return;
        }
    }
    // replace
    int replace = 0;
    int min_use = cache[set][0].use_count;
    for(int i = 1; i < k; i++){
        if(cache[set][i].use_count < min_use){
            replace = i;
            min_use = cache[set][i].use_count;
        }
    }
    cache[set][replace].value = value;
    cache[set][replace].use_count = 0;
    return;
}


int main(){
    cin >> n >> k >> stream_size;
    n_set = n / k;
    struct addr **cache = new addr* [n_set];
    for(int i = 0; i < n_set; i++){
        cache[i] = new addr [k];
        for(int j = 0; j < k; j++){
            cache[i][j].value = -1;
            cache[i][j].use_count = -1;
        }
    }
    
    int address;
    int block;
    int miss_count = 0;
    for(int i = 0;i < stream_size; i++){
        cin >> address;

        if (found(cache, address, i))
            continue;
        
        miss_count ++;
        put(cache, i, address);
        
    }
    cout << "Total Cache Misses:" << miss_count;
    for(int i = 0;i<n_set;i++)
        delete[] cache[i];
    delete [] cache;
    return 0;
}

