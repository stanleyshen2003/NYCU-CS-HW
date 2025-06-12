#include "mpi.h"
#include <fstream>
#include <iostream>

void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr) {
    in >> *n_ptr >> *m_ptr >> *l_ptr;

    int n = *n_ptr, m = *m_ptr, l = *l_ptr;

    // allocate in the form of a single vector
    *a_mat_ptr = new int[n * m];
    *b_mat_ptr = new int[m * l];

    for (int i = 0; i < n * m; ++i) {
        in >> (*a_mat_ptr)[i];
    }

    for (int i = 0; i < m * l; ++i) {
        in >> (*b_mat_ptr)[i];
    }
}


void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // pre-allocate result
    int *result = nullptr;
    if (rank == 0) {
        result = new int[n * l]{};
    }
    
    // Divide the rows of matrix A among processes
    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row, end_row;
    if (rank < remainder){
        start_row = rank * (rows_per_proc + 1);
        end_row = (rank + 1) * (rows_per_proc + 1);
    }
    else {
        start_row = rank * rows_per_proc + remainder;
        end_row = (rank + 1) * rows_per_proc + remainder;
    }

    int local_rows = end_row - start_row;
    int *local_result = new int[local_rows * l];
    for (int i = 0; i < local_rows * l; i++){
        local_result[i] = 0;
    }

    // local computation!
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < l; ++j) {
            for (int k = 0; k < m; ++k) {
                local_result[i * l + j] += a_mat[(start_row + i) * m + k] * b_mat[k * l + j];
            }
        }
    }

    // Gather results to the root process
    int *recvcounts = nullptr;
    int *displs = nullptr;

    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        int start, end;
        for (int i = 0; i < size; ++i) {
            if (i < remainder){
                start = i * (rows_per_proc + 1);
                end = (i + 1) * (rows_per_proc + 1);
            }
            else {
                start = i * rows_per_proc + remainder;
                end = (i + 1) * rows_per_proc + remainder;
            }
            recvcounts[i] = (end - start) * l;
            displs[i] = start * l;
        }
    }

    MPI_Gatherv(local_result, local_rows * l, MPI_INT, result, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Output the result matrix
    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < l; ++j) {
                std::cout << result[i * l + j] << " ";
            }
            std::cout << "\n";
        }

        delete[] result;
        delete[] recvcounts;
        delete[] displs;
    }

    delete[] local_result;
}

void destruct_matrices(int *a_mat, int *b_mat) {
    delete[] a_mat;
    delete[] b_mat;
}
