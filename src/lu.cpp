/*
 *  lu.cpp - Sequential version of LU decomposition
 *  CPA @ M.EIC, 2023
 *  Authors:
 *      Miguel Rodrigues <up201906042@edu.fe.up.pt>
 *      Sérgio Estêvão <up201905680@edu.fe.up.pt>
 */ 
#include <iostream>
#include <memory>

template<typename T>
using matrix_t = T[];


template<typename T> 
void lu(matrix_t<T> A, const std::size_t N)
{
    std::size_t k = 0;
    while (A[k * N + k] != 0 && k <= N) {

        for (std::size_t i = k + 1; i < N; ++i) {
            A[i * N + k] /= A[k * N + k];

            for (std::size_t j = k + 1; j < N; ++j) {
                A[i * N + j] -= A[i * N + k] * A[k * N + j];
            }
        }

        ++k;
    }
}


int main(void)
{
    constexpr std::size_t matrix_size = 4;
    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);

    constexpr auto show = [matrix_size]<typename T>(const matrix_t<T> M) {
        for (std::size_t i = 0; i < matrix_size; ++i) {
            for (std::size_t j = 0; j < matrix_size; ++j) {
                std::cout << M[i * matrix_size + j] << '\t';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    };

    constexpr auto fill = [matrix_size]<typename T>(matrix_t<T> M) {
        for (std::size_t i = 0; i < matrix_size * matrix_size; ++i) {
            M[i] = 1.0 + static_cast<T>(i);
        }
    };

    fill(matrix.get());

    show(matrix.get());
    lu(matrix.get(), matrix_size);
    show(matrix.get());

    return 0;
}
