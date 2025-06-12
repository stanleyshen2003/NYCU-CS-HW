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

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Win win;

    // TODO: MPI init
    
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

    
    

    if (world_rank == 0)
    {
        long long int *counts;
        
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &counts);
        counts[0] = count;
        for (int i = 1; i < world_size; i++)
            counts[i] = -1;
        
        MPI_Win_create(counts, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        bool ready = true;

        do{
            ready = true;
            for (int i = 1; i < world_size; i++)
            {
                MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win);
                if (counts[i] == -1)
                {
                    ready = false;
                    MPI_Win_unlock(i, win);
                    break;
                }
                MPI_Win_unlock(i, win);
                
            }
        } while (!ready);

        long long int global_count = 0;
        for (int i = 0; i < world_size; i++)
            global_count += counts[i];

        count = global_count;

        MPI_Win_free(&win);
        MPI_Free_mem(counts);       
    }
    else
    {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&count, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);
        MPI_Win_free(&win);
    }

    if (world_rank == 0)
    {
        // TODO: handle PI result
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