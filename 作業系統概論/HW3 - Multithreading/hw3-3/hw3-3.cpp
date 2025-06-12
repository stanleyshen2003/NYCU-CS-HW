#include <iostream>
#include <pthread.h>
using namespace std;
int n, global_count = 0, numbercount;
int num_threads;

pthread_mutex_t count_mutex;

// check prime
int is_prime(int num) {
    if (num == 1) return false;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) {
            return 0;
        }
    }
    return 1;
}
// thread funcition
pthread_mutex_t number_mutex;
void* count_primes(void* arg) {
    int thread_id = *((int*)arg);
    int now;
    int local_count = 0;
    int yes;
    while (1) {
        pthread_mutex_lock(&number_mutex);
        if(numbercount<2){
            pthread_mutex_unlock(&number_mutex);
            break;
        }
        now = numbercount--;
        pthread_mutex_unlock(&number_mutex);
        if(is_prime(now)){
            local_count++;
        }
    }
    pthread_mutex_lock(&count_mutex);
    global_count += local_count;
    pthread_mutex_unlock(&count_mutex);
    pthread_exit(NULL);
}
int main(int argc, char** argv) {
    num_threads = stoi(argv[2]);
    pthread_t threads[num_threads];
    int thread_ids[num_threads];
    // Initialize mutex
    pthread_mutex_init(&count_mutex, NULL);
    pthread_mutex_init(&number_mutex, NULL);
    // Read input
    cin >> n;
    numbercount = n;
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, count_primes, (void*)&thread_ids[i]);
    }
    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    // Destroy mutex
    pthread_mutex_destroy(&count_mutex);
    pthread_mutex_destroy(&number_mutex);
    cout << global_count << endl;
    return 0;
}
