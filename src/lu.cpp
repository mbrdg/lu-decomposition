/*
 *  lu.cpp - Sequential version of LU decomposition
 *  CPA @ M.EIC, 2023
 *  Authors:
 *      Miguel Rodrigues <up201906042@edu.fe.up.pt>
 *      Sérgio Estêvão <up201905680@edu.fe.up.pt>
 */ 
#include <iostream>
#include <memory>
#include <random>

template<typename T> 
using matrix_t = T[];
using matrix_size_t = std::size_t;

static constexpr matrix_size_t matrix_size = 3;


template<typename T> 
void lu(matrix_t<T> A, const matrix_size_t N)
{
    for (matrix_size_t k = 0; A[k * N + k] != 0 && k < N; ++k) {

        for (matrix_size_t i = k + 1; i < N; ++i) {
            A[i * N + k] /= A[k * N + k];

            for (matrix_size_t j = k + 1; j < N; ++j) {
                A[i * N + j] -= A[i * N + k] * A[k * N + j];
            }
        }
    }
}


int main(void)
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
    lu(matrix.get(), matrix_size);
    show(matrix.get(), matrix_size);

    return 0;
}
