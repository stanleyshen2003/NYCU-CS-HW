#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

// Compares schedule and oldschedule and prints schedule if different
// Also displays the time in seconds since the first invocation
int fnz (int *schedule, int *oldschedule, int size)
{
    static double starttime = -1.0;
    int diff = 0;

    for (int i = 0; i < size; i++)
       diff |= (schedule[i] != oldschedule[i]);

    if (diff)
    {
       int res = 0;

       if (starttime < 0.0) starttime = MPI_Wtime();

       printf("[%6.3f] Schedule:", MPI_Wtime() - starttime);
       for (int i = 0; i < size; i++)
       {
          printf("\t%d", schedule[i]);
          res += schedule[i];
          oldschedule[i] = schedule[i];
       }
       printf("\n");

       return(res == size-1);
    }
    return 0;
}

int main (int argc, char **argv)
{
    MPI_Win win;
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
       int *oldschedule = malloc(size * sizeof(int));
       // Use MPI to allocate memory for the target window
       int *schedule;
       MPI_Alloc_mem(size * sizeof(int), MPI_INFO_NULL, &schedule);

       for (int i = 0; i < size; i++)
       {
          schedule[i] = 0;
          oldschedule[i] = -1;
       }

       // Create a window. Set the displacement unit to sizeof(int) to simplify
       // the addressing at the originator processes
       MPI_Win_create(schedule, size * sizeof(int), sizeof(int), MPI_INFO_NULL,
          MPI_COMM_WORLD, &win);

       int ready = 0;
       while (!ready)
       {
          // Without the lock/unlock schedule stays forever filled with 0s
          MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
          ready = fnz(schedule, oldschedule, size);
          MPI_Win_unlock(0, win);
       }
       printf("All workers checked in using RMA\n");

       // Release the window
       MPI_Win_free(&win);
       // Free the allocated memory
       MPI_Free_mem(schedule);
       free(oldschedule);

       printf("Master done\n");
    }
    else
    {
       int one = 1;

       // Worker processes do not expose memory in the window
       MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

       // Register with the master
       MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
       MPI_Put(&one, 1, MPI_INT, 0, rank, 1, MPI_INT, win);
       MPI_Win_unlock(0, win);

       printf("Worker %d finished RMA\n", rank);

       // Release the window
       MPI_Win_free(&win);

       printf("Worker %d done\n", rank);
    }

    MPI_Finalize();
    return 0;
}