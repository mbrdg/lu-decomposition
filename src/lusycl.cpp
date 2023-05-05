/*
 *  lusycl.cpp - SYCL version of block LU decomposition
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

static constexpr matrix_size_t matrix_size = 4;
static constexpr block_size_t block_size = 1;


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
void lu(sycl::queue& q,
        matrix_t<T> matrix,
        matrix_t<T> L,
        matrix_t<T> U,
        const matrix_size_t N,
        const block_size_t B)
{
    sycl::buffer buffer { matrix, sycl::range<2>{ N, N } };
    sycl::buffer lower { L, sycl::range<2>{ N, N } };
    sycl::buffer upper { U, sycl::range<2>{ N, N } };

    for (matrix_size_t i = 0; i < N - 1; ++i) {
        q.submit([&](sycl::handler& h) {
            auto reader = buffer.get_access(h);
            auto writer = lower.get_access(h);

            h.parallel_for<class KernelLowerDiagonal>(sycl::nd_range<1>{ N, B }, [=](sycl::nd_item<1> item) {
                const matrix_size_t r = item.get_global_id(0);
                if (r > i) {
                    writer[r][i] = reader[r][i] / reader[i][i];
                }
            });
        });

        q.submit([&](sycl::handler& h) {
            auto reader = buffer.get_access(h);
            auto writer = upper.get_access(h);

            h.parallel_for<class KernelUpperDiagonal>(sycl::nd_range<2>{ sycl::range<2>{ N, N }, sycl::range<2>{ 1, B } }, [=](sycl::nd_item<2> item) {
                const matrix_size_t r = item.get_global_id(0);
                const matrix_size_t c = item.get_global_id(1);

                if (r > i && c >= i) {
                    writer[r][c] = reader[r][c] - reader[i][c] * (reader[r][i] / reader[i][i]);
                } else {
                    writer[r][c] = reader[r][c];
                }
            });
        });

        // FIXME: This is probably not needed
        q.submit([&](sycl::handler& h) {
            auto writer = buffer.get_access(h);
            auto reader = upper.get_access(h);
            h.copy(reader, writer);
        }).wait();
    }
}


int main(void)
{
    sycl::device dev { sycl::default_selector_v };
    sycl::queue q { dev };

    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    make_diagonal_dominant(matrix.get(), matrix_size);

    auto lower = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    auto upper = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);

    const auto start = std::chrono::steady_clock::now();
    lu(q, matrix.get(), lower.get(), upper.get(), matrix_size, block_size);
    const auto end = std::chrono::steady_clock::now();

    show(matrix.get(), matrix_size);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "`lu` took" << ' ' << duration.count() << "ms" << '\n'
              << "matrix size:" << ' ' << matrix_size << '\n'
              << "block size:" << ' ' << block_size << '\n';

    return 0;
}
