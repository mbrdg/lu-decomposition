/*
 *  lublksycl.cpp - Block parallel version of LU decomposition using SYCL
 *  CPA @ M.EIC, 2023
 *  Authors:
 *      Miguel Rodrigues <up201906042@edu.fe.up.pt>
 *      Sérgio Estêvão <up201905680@edu.fe.up.pt>
 */
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>

template <typename T>
using matrix_t = T[];
using matrix_size_t = std::size_t;
using block_size_t = std::size_t;

static constexpr matrix_size_t matrix_size = 7168;
static constexpr block_size_t block_size = 512;


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
void make_diagonal_dominant(matrix_t<T> A, const matrix_size_t N)
{
    std::mt19937 rng(100);
    std::uniform_real_distribution dist(0.0, 1.0);

    for (matrix_size_t i = 0; i < N; ++i) {
        for (matrix_size_t j = 0; j < N; ++j)
            A[i * N + j] = 2.0 * dist(rng) - 1.0;
        A[i * N + i] = dist(rng) + static_cast<T>(N);
    }
}


template <typename T>
void baselu(sycl::queue& q,
            matrix_t<T> A,
            const matrix_size_t N,
            const block_size_t B,
            const matrix_size_t i)
{
    sycl::buffer diagonal_block(A + (i * N + i) * B, sycl::range(B, B));

    for (matrix_size_t ii = 0; ii < B; ++ii) {
        q.submit([&](sycl::handler& h) {
            auto accessor = diagonal_block.get_access(h);
            h.parallel_for(sycl::range(B - ii - 1), [=](sycl::id<1> idx) {
                const matrix_size_t jj = ii + 1 + idx;
                accessor[jj][ii] /= accessor[ii][ii];
            });
        });

        q.submit([&](sycl::handler& h) {
            auto accessor = diagonal_block.get_access(h);
            h.parallel_for(sycl::range(B - ii - 1), [=](sycl::id<1> idx) {
                const matrix_size_t jj = ii + 1 + idx;
                for (matrix_size_t kk = ii + 1; kk < B; ++kk)
                    accessor[jj][kk] -= accessor[jj][ii] * accessor[ii][kk];
            });
        });
    }
}

template <typename T>
void utrsm(sycl::queue& q,
           matrix_t<T> A,
           const matrix_size_t N,
           const block_size_t B,
           const matrix_size_t i,
           const matrix_size_t j)
{
    sycl::buffer diagonal_block(A + (i * N + i) * B, sycl::range(B, B));
    sycl::buffer row_block(A + (i * N + j) * B, sycl::range(B, B));

    q.submit([&](sycl::handler& h) {
        auto diagonal = diagonal_block.template get_access<sycl::access::mode::read>(h);
        auto block = row_block.get_access(h);

        h.parallel_for(sycl::range(B, B), [=](sycl::id<2> idx) {
            const matrix_size_t row = idx[0];
            const matrix_size_t col = idx[1];
            for (matrix_size_t kk = row + 1; kk < B; ++kk)
                block[kk][col] -= diagonal[kk][row] * diagonal[row][col];
        });
    });
}

template <typename T>
void ltrsm(sycl::queue& q,
           matrix_t<T> A,
           const matrix_size_t N,
           const block_size_t B,
           const matrix_size_t i,
           const matrix_size_t j)
{
    sycl::buffer diagonal_block(A + (i * N + i) * B, sycl::range(B, B));
    sycl::buffer col_block(A + (j * N + i) * B, sycl::range(B, B));

    q.submit([&](sycl::handler& h) {
        auto diagonal = diagonal_block.template get_access<sycl::access::mode::read>(h);
        auto block = col_block.get_access(h);

        h.parallel_for(sycl::range(B, B), [=](sycl::id<2> idx) {
            const matrix_size_t row = idx[0];
            const matrix_size_t col = idx[1];
            block[col][row] /= diagonal[row][row];
        });
    });

    q.submit([&](sycl::handler& h) {
        auto diagonal = diagonal_block.template get_access<sycl::access::mode::read>(h);
        auto block = col_block.get_access(h);

        h.parallel_for(sycl::range(B, B), [=](sycl::id<2> idx) {
            const matrix_size_t row = idx[0];
            const matrix_size_t col = idx[1];
            for (matrix_size_t kk = row + 1; kk < B; ++kk)
                block[col][kk] -= block[col][row] * diagonal[row][kk];
        });
    });
}

template <typename T>
void gemm(sycl::queue& q,
          matrix_t<T> A,
          const matrix_size_t N,
          const block_size_t B,
          const matrix_size_t i,
          const matrix_size_t j,
          const matrix_size_t k)
{
    sycl::buffer lower_block(A + (j * N + i) * B, sycl::range(B, B));
    sycl::buffer upper_block(A + (i * N + k) * B, sycl::range(B, B));
    sycl::buffer target_block(A + (j * N + k) * B, sycl::range(B, B));

    q.submit([&](sycl::handler& h) {
        auto lower = lower_block.template get_access<sycl::access::mode::read>(h);
        auto upper = upper_block.template get_access<sycl::access::mode::read>(h);
        auto target = target_block.get_access(h);

        h.parallel_for(sycl::range(B, B), [=](sycl::id<2> idx) {
            const matrix_size_t row = idx[0];
            const matrix_size_t col = idx[1];
            for (matrix_size_t kk = 0; kk < B; ++kk)
                target[row][col] -= lower[row][kk] * upper[kk][col];
        });
    });
}

template <typename T>
void lu(sycl::queue& q,
        matrix_t<T> A,
        const matrix_size_t N,
        const block_size_t B)
{
    const matrix_size_t blocks = N / B;

    for (matrix_size_t i = 0; i < blocks; ++i) {
        baselu(q, A, N, B, i);

        for (matrix_size_t j = i + 1; j < blocks; ++j) {
            utrsm(q, A, N, B, i, j);
        }

        for (matrix_size_t j = i + 1; j < blocks; ++j) {
            ltrsm(q, A, N, B, i, j);

            for (matrix_size_t k = i + 1; k < blocks; ++k) {
                gemm(q, A, N, B, i, j, k);
            }
        }
    }
}

int
main(void) 
{
    sycl::device dev(sycl::default_selector_v);
    sycl::queue q(dev, sycl::property::queue::enable_profiling()); 

    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    make_diagonal_dominant(matrix.get(), matrix_size);

    const auto start = std::chrono::steady_clock::now();
    lu(q, matrix.get(), matrix_size, block_size);
    const auto end = std::chrono::steady_clock::now();

    // WARN: be careful when calling this!
    // show(matrix.get(), matrix_size);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "running on:" << ' ' << q.get_device().get_info<sycl::info::device::name>() << '\n'
              << "`lu` took" << ' ' << duration.count() << "ms" << '\n'
              << "matrix size:" << ' ' << matrix_size << '\n'
              << "block size:" << ' ' << block_size << '\n';

    return 0;
}
