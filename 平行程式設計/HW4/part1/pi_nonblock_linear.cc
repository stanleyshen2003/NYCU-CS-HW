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
    // Initialize MPI
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

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Request request;
        MPI_Send(&count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size - 1];
        // requests = (MPI_Request*)malloc((world_size - 1) * sizeof(MPI_Request));
        long long int *local_count = (long long int *)malloc((world_size - 1) * sizeof(long long int));

        for (int i = 1; i < world_size; i++) {
            MPI_Irecv(&local_count[i - 1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }

        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < world_size - 1; i++) {
            count += local_count[i];
        }
        free(local_count);

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
