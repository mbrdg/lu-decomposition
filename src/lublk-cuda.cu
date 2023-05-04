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

static constexpr matrix_size_t matrix_size = 8192;
static constexpr block_size_t block_size = 128;     // 128 seems to be the better value


__global__ void baselu(double *A, int N, int i){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row > i && col > i){
        A[row * N + col] -= A[row * N + i] * A[i * N + col];
    }
}

__global__ void utrsm(double *A, int N, int B, int i, int j){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row > i && col > j){
        A[row * N + col] -= A[row * N + i] * A[i * N + col];
    }
}

__global__ void ltrsm(double *A, int N, int B, int i, int j){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row > j && col > i){
        A[row * N + col] -= A[row * N + i] * A[i * N + col];
    }
}

__global__ void gemm(double *A, int N, int B, int i, int j, int k){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row > j && col > k){
        A[row * N + col] -= A[row * N + i] * A[i * N + col];
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

   
    for (int i = 0; i < blocks; ++i) { 
        
        baselu<<<grid_size, block_size>>>(gpu_A, N, i);  // LU decomposition on the diagonal block

        cudaDeviceSynchronize();

        for (int j = i + 1; j < blocks; ++j) {
           
            
            utrsm<<<grid_size, block_size>>>(gpu_A, N, B, i, j);    // upper triangular matrix solver
        }

        for (int j = i + 1; j < blocks; ++j) {
            
            ltrsm<<<grid_size, block_size>>>(gpu_A, N, B, i, j);    // lower triangular matrix solver

            for (int k = i + 1; k < blocks; ++k) {
                
                gemm<<<grid_size, block_size>>>(gpu_A, N, B, i, j, k);  // general matrix multiplication
            }
        }
    }
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


int
main(void) 
{

    

    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    make_diagonal_dominant(matrix.get(), matrix_size);

   

    const auto start = std::chrono::steady_clock::now();
    lu(matrix.get(), matrix_size, block_size);
    const auto end = std::chrono::steady_clock::now();


    // show(matrix.get(), matrix_size);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "`lu` took" << ' ' << duration << '\n'
    //           << "matrix size:" << ' ' << matrix_size << '\n'
    //           << "block size:" << ' ' << block_size << '\n';

    return 0;
}
