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
using matrix_size_t = std::size_t;

static constexpr matrix_size_t matrix_size = 4;


template<typename T> 
void lu(matrix_t<T> A, const matrix_size_t N)
{
    for (std::size_t k = 0; A[k * N + k] != 0 && k <= N; ++k) {

        for (std::size_t i = k + 1; i < N; ++i) {
            A[i * N + k] /= A[k * N + k];

            for (std::size_t j = k + 1; j < N; ++j) {
                A[i * N + j] -= A[i * N + k] * A[k * N + j];
            }
        }
    }
}


int main(void)
{
    auto matrix = std::make_unique<matrix_t<double>>(matrix_size * matrix_size);
    for (std::size_t i = 0; i < matrix_size * matrix_size; ++i) {
        matrix[i] = 1.0 + static_cast<double>(i);
    }

    lu(matrix.get(), matrix_size);
    
    for (matrix_size_t i = 0; i < matrix_size; ++i) {
        for (matrix_size_t j = 0; j < matrix_size; ++j) {
            std::cout << matrix[i * matrix_size + j] << '\t';
        }
        std::cout << '\n';
    }

    return 0;
}
