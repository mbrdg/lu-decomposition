// Check the properties of a CUDA device
// Snippet taken from: https://stackoverflow.com/questions/14800009/how-to-get-properties-from-active-cuda-device
#include <cstdio>

int main()
{
    int devc;
    cudaGetDeviceCount(&devc);

    std::printf("No. of devices: %d\n", devc);
    for (int i = 0; i < devc; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        std::printf("Device number: %d\n", i);
        std::printf("\tDevice name: %s\n", props.name);
        std::printf("\tMemory Clock Rate (MHz): %d\n", props.memoryClockRate / 1024);
        std::printf("\tMemory Bus Width (bits): %d\n", props.memoryBusWidth);
        std::printf("\tPeak Memory Bandwidth (GB/s): %.1f\n", 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);
        std::printf("\tTotal global memory (Gbytes) %.1f\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("\tShared memory per block (Kbytes) %.1f\n", props.sharedMemPerBlock / 1024.0);
        std::printf("\tmajor-minor: %d.%d\n", props.major, props.minor);
        std::printf("\tWarp-size: %d\n", props.warpSize);
        std::printf("\tConcurrent kernels: %s\n", props.concurrentKernels ? "yes" : "no");
        std::printf("\tConcurrent computation/communication: %s\n", props.deviceOverlap ? "yes" : "no");
    }
}