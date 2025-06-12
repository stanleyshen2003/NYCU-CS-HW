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

    // TODO: init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int count = 0;
    long long int total_iter;
    unsigned int seed = world_rank;

    if (world_rank > 0)
        total_iter = tosses / world_size;
    else if (world_rank == 0)
        total_iter = tosses / world_size + tosses % world_size;

    for (long long int i = 0; i < total_iter; i++){
        double temp1 = rand_r(&seed) / (double)RAND_MAX; 
        double temp2 = rand_r(&seed) / (double)RAND_MAX; 
        if (temp1 * temp1 + temp2 * temp2 <= 1.0)
            count++;
    }

    // TODO gather info
    int step = 1;
    while (step < world_size) {
        if (world_rank % (2 * step) == 0) {
            // if (world_rank + step < world_size) {
                long long int received_count;
                MPI_Recv(&received_count, 1, MPI_LONG_LONG, world_rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                count += received_count;
            // }
        } else {
            // Send data to another process
            int target = world_rank - step;
            MPI_Send(&count, 1, MPI_LONG_LONG, target, 0, MPI_COMM_WORLD);
            break; // Exit loop after sending
        }
        step *= 2;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 / (double)tosses * (double)count;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
