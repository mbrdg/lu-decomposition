/*
 *  luomp.cu - CUDA version of block LU decomposition
 *  CPA @ M.EIC, 2023
 *  Authors:
 *      Miguel Rodrigues <up201906042@edu.fe.up.pt>
 *      Sérgio Estêvão <up201905680@edu.fe.up.pt>
 */
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

template <typename T>
using matrix_t = T*;
using matrix_size_t = std::size_t;
using block_size_t = std::size_t;

static constexpr matrix_size_t matrix_size = 8192;
static constexpr block_size_t block_size = 128;     // 128 seems to be the better value



template <typename T>
__global__ void utrsm_kernel(T* A, const matrix_size_t N, const block_size_t B, const matrix_size_t j)
{
    const auto [ii, jj] = std::make_pair(threadIdx.x, blockIdx.x * blockDim.y + threadIdx.y);
    const auto [start_row, end_row] = std::make_pair(blockIdx.x * B, blockIdx.x * B + B);
    const auto [start_col, end_col] = std::make_pair(j * B, j * B + B);

    for (auto kk = start_col; kk < end_col; ++kk) {
        if (ii > jj) {
            A[jj * N + kk] -= A[ii * N + kk] * A[jj * N + ii];
        }
    }
}

template <typename T>
__global__ void ltrsm_kernel(T* A, const matrix_size_t N, const block_size_t B, const matrix_size_t i)
{
    const auto [ii, jj] = std::make_pair(threadIdx.x, blockIdx.x * blockDim.y + threadIdx.y);
    const auto [start_row, end_row] = std::make_pair(i * B, i * B + B);
    const auto [start_col, end_col] = std::make_pair(blockIdx.x * B, blockIdx.x * B + B);

    for (auto kk = start_row; kk < end_row; ++kk) {
        if (jj > ii) {
            A[jj * N + kk] -= A[ii * N + kk] * A[jj * N + ii];
        }
    }
}

template <typename T>
__global__ void gemm_kernel(T* A, const matrix_size_t N, const block_size_t B, const matrix_size_t i, const matrix_size_t j, const matrix_size_t k)
{
    const auto [ii, jj] = std::make_pair(threadIdx.x, blockIdx.x * blockDim.y + threadIdx.y);
    const auto [start_row, end_row] = std::make_pair(i * B, i * B + B);
    const auto [start_col, end_col] = std::make_pair(j * B, j * B + B);
    const auto [start_res, end_res] = std::make_pair(blockIdx.x * B, blockIdx.x * B + B);

    for (auto kk = start_col; kk < end_col; ++kk) {
        for (auto rr = start_res; rr < end_res; ++rr) {
            A[rr * N + kk] -= A[ii * N + kk] * A[rr * N + ii];
        }
    }
}

template <typename T>
void lu_cuda(matrix_t<T> A, const matrix_size_t N, const block_size_t B)
{
    const dim3 grid(N/B, 1, 1);
    const dim3 block(B, B, 1);
    const auto blocks = static_cast<int>(N / B);

    for (int i = 0; i < blocks; ++i) {
        // LU decomposition on the diagonal block
        baselu<<<1, B>>>(A, N, B, i);
        cudaDeviceSynchronize();

        for (int j = i + 1; j < blocks; ++j) {
            utrsm_kernel<<<grid, block>>>(A, N, B, i);
            cudaDeviceSynchronize();
            ltrsm_kernel<<<grid, block>>>(A, N, B, j);
            cudaDeviceSynchronize();
            gemm_kernel<<<grid, block>>>(A, N,B, i, j, i);
            cudaDeviceSynchronize();
            }
    }
}

template <typename T>
__global__ void make_diagonal_dominant(T* A, const matrix_size_t N) {
    std::mt19937 rng(100);
    std::uniform_real_distribution dist(0.0, 1.0);

    const matrix_size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        for (matrix_size_t j = 0; j < N; ++j) {
            A[tid * N + j] = 2.0 * dist(rng) - 1.0;
        }
        A[tid * N + tid] = dist(rng) + static_cast<T>(N);
    }
}

template <typename T>
__global__ void show(const T* A, const matrix_size_t N) {
    for (matrix_size_t i = 0; i < N; ++i) {
        for (matrix_size_t j = 0; j < N; ++j) {
            printf("%.2f\t", A[i * N + j]);
        }
        printf("\n");
    }
}

int main(void) 
{
    

    auto h_matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    const dim3 grid(matrix_size / block_size + 1);
    const dim3 block(matrix_size);
    make_diagonal_dominant<<<grid, block>>>(h_matrix.get(), matrix_size);

    auto d_matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    cudaMemcpy(d_matrix.get(), h_matrix.get(), matrix_size * matrix_size * sizeof(double), cudaMemcpyHostToDevice);

    const auto start = std::chrono::steady_clock::now();
    lu_cuda(d_matrix.get(), matrix_size, block_size);
    const auto end = std::chrono::steady_clock::now();

    cudaMemcpy(h_matrix.get(), d_matrix.get(), matrix_size * matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
    // show(h_matrix.get(), matrix_size);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "`lu` took" << ' ' << duration.count() << "ms\n"
              << "matrix size:" << ' ' << matrix_size << '\n'
              << "block size:" << ' ' << block_size << '\n';

    return 0;
}

