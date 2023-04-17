// Hello world in CUDA
#include <stdio.h>


__global__ void cuda_say_hello() 
{
    printf("Hello World!\n"
           "This is being executed from the GPU!\n");
}

int main() 
{
    cuda_say_hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
