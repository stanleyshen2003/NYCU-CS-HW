#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int count = 0, total_iter;
    if (world_rank > 0)
        total_iter = tosses / world_size;
    else if (world_rank == 0)
        total_iter = tosses / world_size + tosses % world_size;

    unsigned int seed = world_rank;

    for (long long int i = 0; i < total_iter; i++) {
        double x = rand_r(&seed) / (double)RAND_MAX;
        double y = rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) {
            count++;
        }
    }
    


    // TODO: use MPI_Gather
    long long int *counts = NULL;
    if (world_rank == 0) {
        counts = new long long int[world_size];
    }

    MPI_Gather(&count, 1, MPI_LONG_LONG, counts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);


    if (world_rank == 0)
    {
        // TODO: PI result
        long long int global_count = 0;
        for (int i = 0; i < world_size; i++) {
            global_count += counts[i];
        }

        pi_result = 4.0 / (double)tosses * (double)global_count;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    delete[] counts;
    
    MPI_Finalize();
    return 0;
}
