#include <iostream>
#include <cstdint>
#include <vector>
#include <pthread.h>
#include <cstring>
using namespace std;

int n, m, total_thread=1, shiftamt;
vector<uint64_t> subsets;
uint64_t one = static_cast<uint64_t>(1), global_count = 0;
pthread_mutex_t mutex;

int localcount[8];

void solving(int index, uint64_t current,int i) {
    if (index == m) {
        if (current == (one << n) - 1){
            localcount[i]++;
        } 
    } else {
        solving(index + 1, current, i);
        solving(index + 1, current | subsets[index], i);
    }
}
void *solve(void* arg){
    uint64_t * threadnum = (uint64_t *)arg;
    uint64_t current=0,tmp = *threadnum;
    int count = 0;
    while(tmp){
        if(tmp & 1)
            current |= subsets[count];
        tmp = tmp >> 1;
        count++;
    }
    int threadid = static_cast<int>(*threadnum);
    
    solving(shiftamt, current,threadid);
    
    pthread_exit(0);
}



int getlog2(int num){
    int log = 0;
    while(num!=1){
        log++;
        num/=2;
    }
    return log;
}


int main(int argc, char ** argv) {
    cin >> n >> m;
    for(int i=1;i<argc-1;i++){
        if(!strcmp(argv[i], "-t"))
        total_thread = (int)(argv[2][0] - '0');

    }
    subsets.resize(m);
    for (int i = 0; i < m; i++) {
        int p, temp;
        cin >> p;
        for (int j = 0; j < p; j++) {
            cin >> temp;
            subsets[i] |= (one << temp);
        }
    }
    shiftamt = getlog2(total_thread);
    
    pthread_t threads[total_thread];
    int64_t numbers[8];
    for (int i = 0; i < total_thread; ++i) {
        numbers[i] = i;
        pthread_create(&threads[i], nullptr, solve, &numbers[i]);
    }

    for (int i = 0; i < total_thread; ++i) {
        pthread_join(threads[i], nullptr);        
    }

    for(int i=0;i<total_thread;i++){
        global_count += localcount[i];
    }
    
    cout << global_count << endl;
    return 0;
}
