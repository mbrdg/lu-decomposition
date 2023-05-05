/*
 *  luomp.cpp - OpenMP version of block LU decomposition
 *  CPA @ M.EIC, 2023
 *  Authors:
 *      Miguel Rodrigues <up201906042@edu.fe.up.pt>
 *      Sérgio Estêvão <up201905680@edu.fe.up.pt>
 */
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <utility>


template <typename T>
using matrix_t = T[];
using matrix_size_t = std::size_t;
using block_size_t = std::size_t;

static constexpr matrix_size_t matrix_size = 8;
static constexpr block_size_t block_size = 2;     // 128 seems to be the better value


__global__ void baselu(double *A, const matrix_size_t N,
            const block_size_t B,
            const matrix_size_t i){
    const auto [start, end] = std::make_pair(i * B, i * B + B);

    for (auto ii = start; A[ii * N + ii] != 0 && ii < end - 1; ++ii) {    
        for (auto jj = ii + 1; jj < end; ++jj) {
            A[jj * N + ii] /= A[ii * N + ii];

            for (auto kk = ii + 1; kk < end; ++kk) {
                A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
            }
        }
    }
}


__global__ void row_col_solver(double *A, const matrix_size_t N, const matrix_size_t i){
    
    
    if (threadIdx.x < blockDim.x/2) { // utrsm
        const auto j = threadIdx.x + i + 1;
        printf("j: %d\n", j);

        const auto [start_row, end_row] = std::make_pair(i * blockDim.x, i * blockDim.x + blockDim.x);
        const auto [start_col, end_col] = std::make_pair(j * blockDim.x, j * blockDim.x + blockDim.x);

        for (auto ii = start_row; ii < end_row - 1; ++ii) {
            for (auto jj = ii + 1; jj < blockDim.x; ++jj) {
                for (auto kk = start_col; kk < end_col; ++kk) {
                    A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
                }
            }
        }
    }
    else if(threadIdx.x < blockDim.x){ // ltrsm
        const auto j = threadIdx.x - blockDim.x/2 + i + 1;
        printf("j: %d\n", j);
        const auto [start_row, end_row] = std::make_pair(i * blockDim.x, i * blockDim.x + blockDim.x);
        const auto [start_col, end_col] = std::make_pair(j * blockDim.x, j * blockDim.x + blockDim.x);

        printf("start_row: %d\n", start_row);
        printf("end_row: %d\n", end_row);
        printf("start_col: %d\n", start_col);
        printf("end_col: %d\n", end_col);

        for (auto ii = start_row; A[ii * N + ii] != 0 && ii < end_row; ++ii) {
        
            for (auto jj = start_col; jj < end_col; ++jj) {
                A[jj * N + ii] /= A[ii * N + ii];

                for (auto kk = ii + 1; kk < end_row; ++kk) {
                    A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
                }
            }
        }
    }


}

__global__ void gemm(double *A, int N, int i){
    //if(threadIdx.x > ) return;

    const auto j = blockIdx.x + i + 1;
    const auto k = threadIdx.x + i + 1;

    const auto [si, ei] = std::make_pair(i * blockDim.x, i * blockDim.x + blockDim.x);
    const auto [sj, ej] = std::make_pair(j * blockDim.x, j * blockDim.x + blockDim.x);
    const auto [sk, ek] = std::make_pair(k * blockDim.x, k * blockDim.x + blockDim.x);

    for (auto ii = si; ii < ei; ++ii) {
        for (auto jj = sj; jj < ej; ++jj) {
            for (auto kk = sk; kk < ek; ++kk) {
                A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
            }
        }
    }
}


template <typename T>
void lu(matrix_t<T> A, const matrix_size_t N, const block_size_t B)
{
    const int blocks = static_cast<int>(N / B);

    double* gpu_A;
    cudaMalloc((void **)&gpu_A, N * N * sizeof(double));
    cudaMemcpy(gpu_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 grid_size(blocks);
    dim3 block_size(B);

   
    for (int i = 0; i < 1; ++i) { 
        baselu<<<1, 1>>>(gpu_A, N, B, i);  // LU decomposition on the diagonal block
        

        cudaDeviceSynchronize();

          
        row_col_solver<<<1, blocks*2-2>>>(gpu_A, N, i);  // solve the rows and columns of the diagonal block

        cudaDeviceSynchronize();

        // gemm<<<blocks-1, blocks-1>>>(gpu_A, N, i);  // general matrix multiplication

        // cudaDeviceSynchronize();

        
    }


    cudaMemcpy(A, gpu_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(gpu_A);
}

template<typename T>
void make_diagonal_dominant(matrix_t<T> A, const matrix_size_t N){
    std::mt19937 rng(100);
    std::uniform_real_distribution dist(0.0, 1.0);

    for (matrix_size_t i = 0; i < N; ++i) {
        for (matrix_size_t j = 0; j < N; ++j)
            A[i * N + j] = 2.0 * dist(rng) - 1.0;
        A[i * N + i] = dist(rng) + static_cast<T>(N);
    }
}

template<typename T>
void show(matrix_t<T> A, const matrix_size_t N, std::ostream& out = std::cout) {
    for (matrix_size_t i = 0; i < N; ++i) {
        for (matrix_size_t j = 0; j < N; ++j) {
            out << std::fixed << A[i * N + j] << '\t';
        }
        out << '\n';
    }
}


int
main(void) 
{

    

    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    make_diagonal_dominant(matrix.get(), matrix_size);

   

    const auto start = std::chrono::steady_clock::now();
    lu(matrix.get(), matrix_size, block_size);
    const auto end = std::chrono::steady_clock::now();

    show(matrix.get(), matrix_size, std::cout);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "`lu` took" << ' ' << duration.count() << '\n'
              << "matrix size:" << ' ' << matrix_size << '\n'
              << "block size:" << ' ' << block_size << '\n';

    return 0;
}
