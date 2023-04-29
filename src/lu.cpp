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


template<typename T, std::size_t N> 
void lu(matrix_t<T> A)
{
    std::size_t k = 0;
    while (A[k * N + k] != 0 && k <= N) {
        
        for (std::size_t i = k + 1; i < N; ++i) {
            A[i * N + k] /= A[k * N + k];
        }

        for (std::size_t i = k + 1; i < N; ++i) {
            A[i * N + i] -= A[i * N + k] * A[k * N + i];
        }

        ++k;
    }
}


int
main(void)
{

    constexpr std::size_t matrix_size = 2;
    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);

    matrix[0] = 4.0;
    matrix[1] = 3.0;
    matrix[2] = 6.0;
    matrix[3] = 3.0;

    lu<double, matrix_size>(matrix.get());

    for (std::size_t i = 0; i < matrix_size; ++i) {
        for (std::size_t j = 0; j < matrix_size; ++j) {
            std::cout << matrix[i * matrix_size + j] << '\t';
        }
        std::cout << '\n';
    }

    return 0;
}