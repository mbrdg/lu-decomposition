/*
 *  lucuda.cpp - CUDA version of block LU decomposition
 *  CPA @ M.EIC, 2023
 *  Authors:
 *      Miguel Rodrigues <up201906042@edu.fe.up.pt>
 *      Sérgio Estêvão <up201905680@edu.fe.up.pt>
 */
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
static constexpr block_size_t block_size = 512;     // 128 seems to be the better value

template <typename T>
__global__ void baselu(T *A, const matrix_size_t N,
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

template <typename T>
__global__ void row_col_solver(T *A, const matrix_size_t N, const int block_size, const matrix_size_t i) {

    if(threadIdx.x > blockDim.x) return;


    if (blockIdx.x * 2 < gridDim.x) { // utrsm
        
        const auto [start_row, end_row] = std::make_pair(i * blockDim.x, (i + 1) * blockDim.x); //matrix we are operating on
        const auto [start_col, end_col] = std::make_pair((blockIdx.x + i + 1)* blockDim.x, (blockIdx.x + i + 2) * blockDim.x); //diagonal matrix i

        const auto ii = threadIdx.x + start_row;
        for (auto jj = ii + 1; jj < blockDim.x; ++jj) {
            for (auto kk = start_col; kk < end_col; ++kk) {
                A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
            }
        }
    } else if (blockIdx.x < gridDim.x) { // ltrsm

        const auto block_id_ltrsm = blockIdx.x - gridDim.x/2;

        const auto [start_row, end_row] = std::make_pair(i * blockDim.x, (i+1) * blockDim.x);
        const auto [start_col, end_col] = std::make_pair((block_id_ltrsm+i+1) * blockDim.x, (block_id_ltrsm+i+2) * blockDim.x);

        const auto ii = threadIdx.x + start_row;
        for (auto jj = start_col; jj < end_col; ++jj) {
            A[jj * N + ii] /= A[ii * N + ii];

            for (auto kk = ii + 1; kk < end_row; ++kk) {
                A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
            }
        }
        
    }


}

template <typename T>
__global__ void gemm(T *A, int N, const int num_side_blocks, int i) {
    if(threadIdx.x > blockDim.x) return;

    const auto block_row = (blockIdx.x / (num_side_blocks-i-1)) + 1+i; //j
    const auto block_col = (blockIdx.x % (num_side_blocks-i-1)) + 1+i; //k
    // if(threadIdx.x == 0)
    //     printf("block row: %d\nblock col: %d\n", block_row*blockDim.x, block_col*blockDim.x);

    const auto [si, ei] = std::make_pair(i * blockDim.x, i * blockDim.x + blockDim.x);
    const auto [sj, ej] = std::make_pair(block_row * blockDim.x, block_row * blockDim.x + blockDim.x);
    const auto [sk, ek] = std::make_pair(block_col * blockDim.x, block_col * blockDim.x + blockDim.x);

    const auto ii = si + threadIdx.x;

    for (auto jj = sj; jj < ej; ++jj) {
        for (auto kk = sk; kk < ek; ++kk) {
            A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
        }
    }
    
    
   
    

    
}


template <typename T>
void lu(matrix_t<T> A, const matrix_size_t N, const block_size_t B)
{
    const block_size_t blocks = static_cast<int>(N / B);

    T* gpu_A;
    cudaMalloc((void **) &gpu_A, N * N * sizeof(T));
    cudaMemcpy(gpu_A, A, N * N * sizeof(T), cudaMemcpyHostToDevice);

    dim3 grid_size(blocks);
    dim3 block_size(B);
   
    for (block_size_t i = 0; i < blocks; ++i) { 
        baselu<<<1, 1>>>(gpu_A, N, B, i);  // LU decomposition on the diagonal block
        cudaDeviceSynchronize();

        int row_col_blocks = (blocks-1) * 2; 
        
        row_col_solver<<<row_col_blocks, B>>>(gpu_A, N, B, i);  // solve the rows and columns of the diagonal block
        cudaDeviceSynchronize();

        int gemm_blocks = (blocks - i - 1) * (blocks - i - 1);

        //printf("i: %d\ngemm blocks: %d\n", i, gemm_blocks);
        gemm<<<gemm_blocks, B>>>(gpu_A, N, blocks, i);  // general matrix multiplication
        cudaDeviceSynchronize();
        
        

        
    }


    cudaMemcpy(A, gpu_A, N * N * sizeof(T), cudaMemcpyDeviceToHost);
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

    // WARN: be careful when calling this!
    //show(matrix.get(), matrix_size);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "`lu` took" << ' ' << duration.count() << "ms" << '\n'
              << "matrix size:" << ' ' << matrix_size << '\n'
              << "block size:" << ' ' << block_size << '\n';

    return 0;
}
