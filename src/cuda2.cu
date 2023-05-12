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
static constexpr block_size_t block_size = 2;


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

template <typename T>
void show(matrix_t<T> A, const matrix_size_t N, std::ostream& out = std::cout)
{
    for (matrix_size_t i = 0; i < N; ++i) {
        for (matrix_size_t j = 0; j < N; ++j) {
            out << std::fixed << A[i * N + j] << '\t';
        }

        out << '\n';
    }
}


template <typename T>
__global__ baselu(T* A, const matrix_size_t N, const block_size_t B)
{
    const matrix_size_t jj = ii + 1 + threadIdx.x;

    for (matrix_size_t ii = 0; ii < B; ++ii) {
        A[jj * N + ii] /= A[ii * N + ii];
        __syncthreads();
        for (matrix_size_t kk = ii + 1; kk < B; ++k)
            A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + k];
    }
}

template <typename T>
__global__ utrsm(T* A, 
                 const matrix_size_t N,
                 const block_size_t B,
                 const matrix_size_t i)
{
    const matrix_size_t jj = blockIdx.x + 1 + threadIdx.x;

    for (matrix_size_t kk = threadIdx.x + 1; kk < B; ++kk)
        A[kk * N + jj] -= 
}

template <typename T>
void lu(matrix_t<T> A, const matrix_size_t N, const block_size_t B)
{
    const matrix_size_t blocks = N / B;
    T* gpu_A;
    cudaMalloc((void **) &gpu_A, N * N * sizeof(T));
    cudaMemcpy(gpu_A, A, N * N * sizeof(T), cudaMemcpyHostToDevice);

    for (matrix_size_t i = 0; i < blocks; ++i) {
        baselu<<<1, B>>>(gpu_A, N, B, i);

        utrsm<<<blocks - (i + 1), B>>>(gpu_A, N, B, i);
        for (matrix_size_t j = i + 1; j < blocks; ++j) {
        }

        for (matrix_size_t j = i + 1; j < blocks; ++j) {
            ltrsm(gpu_A, N, B, i, j);

            for (matrix_size_t k = i + 1; k < blocks; ++k) {
                gemm(gpu_A, N, B, i, j, k);
            }
        }
    }
}

int main(void)
{
    const matrix_size_t blocks = N / B;
    T* gpu_A;
    cudaMalloc((void **) &gpu_A, N * N * sizeof(T));
    cudaMemcpy(gpu_A, A, N * N * sizeof(T), cudaMemcpyHostToDevice);


}