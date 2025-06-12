#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

long long int number_in_circle = 0;
long long int total_tosses;
pthread_mutex_t mutex;

void* monte_carlo(void* thread_tosses) {
    long long int tosses = *((long long int*)thread_tosses);
    long long int local_in_circle = 0;
    double x, y;
    unsigned int seed = 0;
    // printf("%d", RAND_MAX);
    for (long long int i = 0; i < tosses; i++) {
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_in_circle++;
        }
    }

    pthread_mutex_lock(&mutex);
    number_in_circle += local_in_circle;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num_threads> <total_tosses>\n", argv[0]);
        return 1;
    }

    total_tosses = atoll(argv[2]);
    int num_threads = atoi(argv[1]);

    pthread_t threads[num_threads];
    long long int tosses_per_thread = total_tosses / num_threads;
    long long int last_tosses = tosses_per_thread + total_tosses % num_threads;

    // initialize mutex
    pthread_mutex_init(&mutex, NULL);

    // thread create
    for (int i = 0; i < num_threads - 1; i++) {
        pthread_create(&threads[i], NULL, monte_carlo, &tosses_per_thread);
    }
    pthread_create(&threads[num_threads-1], NULL, monte_carlo, &last_tosses);

    // join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Final PI estimation
    double pi = 4 * (number_in_circle / (double)total_tosses);
    printf("%f\n", pi);

    // Clean up
    pthread_mutex_destroy(&mutex);

    return 0;
}
