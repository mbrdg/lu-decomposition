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


template<typename T>
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
void baselu(matrix_t<T> A,
            const matrix_size_t N,
            const block_size_t B,
            const matrix_size_t i)
{
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
void utrsm(matrix_t<T> A,
           const matrix_size_t N,
           const block_size_t B,
           const matrix_size_t i,
           const matrix_size_t j)
{
    const auto [start_row, end_row] = std::make_pair(i * B, i * B + B);
    const auto [start_col, end_col] = std::make_pair(j * B, j * B + B);

    for (auto ii = start_row; ii < end_row - 1; ++ii) {
        for (auto jj = ii + 1; jj < B; ++jj) {
            for (auto kk = start_col; kk < end_col; ++kk) {
                A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
            }
        }
    }
}

template <typename T>
void ltrsm(matrix_t<T> A,
           const matrix_size_t N,
           const block_size_t B,
           const matrix_size_t i,
           const matrix_size_t j)
{
    const auto [start_row, end_row] = std::make_pair(i * B, i * B + B);
    const auto [start_col, end_col] = std::make_pair(j * B, j * B + B);

    for (auto ii = start_row; A[ii * N + ii] != 0 && ii < end_row; ++ii) {
        
        for (auto jj = start_col; jj < end_col; ++jj) {
            A[jj * N + ii] /= A[ii * N + ii];

            for (auto kk = ii + 1; kk < end_row; ++kk) {
                A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
            }
        }
    }
}

template <typename T>
void gemm(matrix_t<T> A,
          const matrix_size_t N,
          const block_size_t B,
          const matrix_size_t i,
          const matrix_size_t j,
          const matrix_size_t k)
{
    const auto [si, ei] = std::make_pair(i * B, i * B + B);
    const auto [sj, ej] = std::make_pair(j * B, j * B + B);
    const auto [sk, ek] = std::make_pair(k * B, k * B + B);

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
    const int max_task_priority = omp_get_max_task_priority();
    const int blocks = static_cast<int>(N / B);

    #pragma omp parallel
    #pragma omp single nowait
    for (int i = 0; i < blocks; ++i) { 
        #pragma omp task \
            default(none) shared(A, N, B) firstprivate(i) \
            depend(inout: A[(i * N + i) * B]) \
            priority(max_task_priority)
        baselu(A, N, B, i);  // LU decomposition on the diagonal block

        for (int j = i + 1; j < blocks; ++j) {
            #pragma omp task \
                default(none) shared(A, N, B) firstprivate(i, j) \
                depend(in: A[(i * N + i) * B]) depend(inout: A[(i * N + j) * B]) \
                priority(std::max(0, max_task_priority - j + i))
            utrsm(A, N, B, i, j);    // upper triangular matrix solver
        }

        for (int j = i + 1; j < blocks; ++j) {
            #pragma omp task \
                default(none) shared(A, N, B) firstprivate(i, j) \
                depend(in: A[(i * N + i) * B]) depend(inout: A[(j * N + i) * B]) \
                priority(std::max(0, max_task_priority - j + i))
            ltrsm(A, N, B, i, j);    // lower triangular matrix solver

            for (int k = i + 1; k < blocks; ++k) {
                #pragma omp task \
                    default(none) shared(A, N, B) firstprivate(i, j, k) \
                    depend(in: A[(i * N + k) * B], A[(j * N + i) * B]) \
                    depend(inout: A[(j * N + k) * B]) \
                    priority(std::max({0, max_task_priority - j + i, max_task_priority - k + i}))
                gemm(A, N, B, i, j, k);  // general matrix multiplication
            }
        }
    }
}


int main(void) 
{
    const auto make_diagonal_dominant = []<typename T>(matrix_t<T> A, const matrix_size_t N) {
        std::mt19937 rng(100);
        std::uniform_real_distribution dist(0.0, 1.0);

        for (matrix_size_t i = 0; i < N; ++i) {
            for (matrix_size_t j = 0; j < N; ++j)
                A[i * N + j] = 2.0 * dist(rng) - 1.0;
            A[i * N + i] = dist(rng) + static_cast<T>(N);
        }
    };

    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    make_diagonal_dominant(matrix.get(), matrix_size);

    const auto start = std::chrono::steady_clock::now();
    lu(matrix.get(), matrix_size, block_size);
    const auto end = std::chrono::steady_clock::now();

    // WARN: be careful when calling this!
    // show(matrix.get(), matrix_size);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "`lu` took" << ' ' << duration.count() << "ms" << '\n'
              << "matrix size:" << ' ' << matrix_size << '\n'
              << "block size:" << ' ' << block_size << '\n';

    return 0;
}
