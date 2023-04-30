/*
 *  lublk.cpp - Sequential version of block LU decomposition
 *  CPA @ M.EIC, 2023
 *  Authors:
 *      Miguel Rodrigues <up201906042@edu.fe.up.pt>
 *      Sérgio Estêvão <up201905680@edu.fe.up.pt>
 */
#include <iostream>
#include <memory>
#include <random>
#include <utility>

template <typename T>
using matrix_t = T[];
using matrix_size_t = std::size_t;
using block_size_t = std::size_t;

static constexpr matrix_size_t matrix_size = 3;
static constexpr block_size_t block_size = 1;


template <typename T>
void lu(matrix_t<T> A, const matrix_size_t N, const block_size_t B)
{
    const auto baselu = [&A, &N, &B](const matrix_size_t i) {
        const auto [start, end] = std::make_pair(i * B, i * B + B);

        for (auto ii = start; A[ii * N + ii] != 0 && ii < end - 1; ++ii) {

            for (auto jj = ii + 1; jj < end; ++jj) {
                A[jj * N + ii] /= A[ii * N + ii];

                for (auto kk = ii + 1; kk < end; ++kk) {
                    A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
                }
            }
        }
    };

    const auto utrms = [&A, &N, &B](const matrix_size_t i, const matrix_size_t j) {
        const auto [start_row, end_row] = std::make_pair(i * B, i * B + B);
        const auto [start_col, end_col] = std::make_pair(j * B, j * B + B);

        for (auto ii = start_row; ii < end_row - 1; ++ii) {
            for (auto jj = ii + 1; j < B; ++jj) {
                for (auto kk = start_col; kk < end_col; ++kk) {
                    A[jj * N + kk] -= A[jj * N + ii] * A[ii * N + kk];
                }
            }
        }
    };

    const auto ltrms = [&A, &N, &B](const matrix_size_t i, const matrix_size_t j) {
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
    };

    const auto gemm = [&A, &N, &B](const matrix_size_t i, const matrix_size_t j, const matrix_size_t k) {
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
    };

    const matrix_size_t blocks = N / B;

    for (matrix_size_t i = 0; i < blocks; ++i) {        
        baselu(i);  // LU decomposition on the diagonal block

        for (matrix_size_t j = i + 1; j < blocks; ++j) {
            utrms(i, j);    // Upper Triangular matrix solver
        }

        for (matrix_size_t j = i + 1; j < blocks; ++j) {
            ltrms(i, j);    // Lower Triangular matrix solver

            for (matrix_size_t k = i + 1; k < blocks; ++k) {
                gemm(i, j, k);  // General Matrix Multiplication
            }
        }
    }
}



int
main(void) 
{
    const auto make_diagonal_dominant = []<typename T>(matrix_t<T> A, const matrix_size_t N) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution dist(0.0, 1.0);

        for (matrix_size_t i = 0; i < N; ++i) {
            for (matrix_size_t j = 0; j < N; ++j)
                A[i * N + j] = 2.0 * dist(rng) - 1.0;
            A[i * N + i] = dist(rng) + static_cast<double>(N);
        }
    };

    const auto show = []<typename T>(matrix_t<T> A, const matrix_size_t N, std::ostream& out = std::cout) {
        for (matrix_size_t i = 0; i < N; ++i) {
            for (matrix_size_t j = 0; j < N; ++j) {
                out << std::fixed << A[i * N + j] << '\t';
            }
            out << '\n';
        }
    };

    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    make_diagonal_dominant(matrix.get(), matrix_size);
    lu(matrix.get(), matrix_size, block_size);
    show(matrix.get(), matrix_size);

    return 0;
}