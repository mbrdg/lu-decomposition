/*
 *  lusycl.cpp - Base parallel version of LU decomposition using SYCL
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

static constexpr matrix_size_t matrix_size = 8192;


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
void lu(sycl::queue& q, matrix_t<T> A, const matrix_size_t N)
{
    sycl::buffer buffer(A, sycl::range<2>(N, N));

    for (matrix_size_t i = 0; i < N; ++i) {
        q.submit([&](sycl::handler& h) {
            auto accessor = buffer.get_access(h);
            h.parallel_for(sycl::range(N - i - 1), [=](sycl::id<1> idx) {
                const matrix_size_t j = i + 1 + idx;
                accessor[j][i] /= accessor[i][i];
            });
        });

        q.submit([&](sycl::handler& h) {
            auto accessor = buffer.get_access(h);
            h.parallel_for(sycl::range(N - i - 1), [=](sycl::id<1> idx) {
                const matrix_size_t j = i + 1 + idx;
                for (matrix_size_t k = i + 1; k < N; ++k)
                    accessor[j][k] -= accessor[j][i] * accessor[i][k];
            });
        });
    }
}


int main(void)
{
    sycl::device dev(sycl::default_selector_v);
    sycl::queue q(dev);

    std::clog << "running on:" << ' ' << q.get_device().get_info<sycl::info::device::name>() << '\n';

    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    make_diagonal_dominant(matrix.get(), matrix_size);

    const auto start = std::chrono::steady_clock::now();
    lu(q, matrix.get(), matrix_size);
    const auto end = std::chrono::steady_clock::now();

    // show(matrix.get(), matrix_size);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::clog << "`lu` took" << ' ' << duration.count() << "ms" << '\n'
              << "matrix size:" << ' ' << matrix_size << '\n';

    return 0;
}
