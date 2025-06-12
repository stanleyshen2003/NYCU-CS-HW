#include <mpi.h>
#include <fstream>
#include <iostream>

// *********************************************
// ** ATTENTION: YOU CANNOT MODIFY THIS FILE. **
// *********************************************

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from in
//
// in:        input stream of the matrix file
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr);

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat);

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat);

int main (int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " FILE\n";
        return 1;
    }

    MPI_Init(NULL, NULL);
    double start_time = MPI_Wtime();

    int n, m, l;
    int *a_mat, *b_mat;
    std::ifstream in(argv[1]);
    construct_matrices(in, &n, &m, &l, &a_mat, &b_mat);
    matrix_multiply(n, m, l, a_mat, b_mat);
    destruct_matrices(a_mat, b_mat);
    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
        std::cout << "MPI running time: " << end_time - start_time << " Seconds\n";

    MPI_Finalize();
    return 0;
}
