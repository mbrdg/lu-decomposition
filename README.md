# LU Decomposition

LU decomposition (without partial pivoting)
for solving linear systems using OpenMP, CUDA and SYCL.

## Dependencies

### Hardware dependencies

Since this project uses CUDA, which is only available for nvidia gpus, 
you are required to have a compatible gpu, at least to run the CUDA and SYCL targets.

We also assume that you have the corresponding drivers installed.
In Linux, you can use `nvidia-smi` to test if your drivers are installed correctly.

### Software dependencies

1. [GNU Make](https://www.gnu.org/software/make/), as the build system.
2. C++ compiler compatible with C++20 and OpenMP 4.5 or later.
    1. We only tested for [`g++`](https://gcc.gnu.org/) but other compilers should work.
3. [`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/), the nvidia CUDA compiler.
4. [`icpx`](https://intel.github.io/llvm-docs/), the Intel's Data Parallel C++ compiler wich implements the SYCL specification.
    1. Follow this [guide](https://developer.codeplay.com/products/oneapi/nvidia/2023.1.0/guides/get-started-guide-nvidia)
       on how to install the CUDA backend extension.
    2. Our implementation uses the [SYCL 2020 specification](https://www.khronos.org/files/sycl/sycl-2020-reference-guide.pdf).

#### Note

The installation of these dependencies can vary from system to system.
It is up to you to figure out how to install and execute them in your system.

## Usage

First, if you wish to build the `lusycl` target,
you must [set up the environment](https://developer.codeplay.com/products/oneapi/nvidia/2023.1.0/guides/get-started-guide-nvidia#set-up-your-environment).

```console
. /opt/intel/oneapi/setvars.sh --include-intel-llvm
```

After that the following commands can be used to build each of the different targets.

```console
make [all | lu | lublk | luomp | lucuda | lusycl]   # Builds a target
./bin/lu.out      # Serial LU decomposition
./bin/lublk.out   # Block-based serial LU decomposition
./bin/luomp.out   # Block-based parallel LU decomposition using OpenMP
./bin/lucuda.out  # Block-based parallel LU decomposition using CUDA
./bin/lusycl.out  # Block-based parallel LU decomposition usin SYCL
make clean        # Removes all the produced executables
```

## Authors

- [Miguel Rodrigues](mailto:up201906042@edu.fe.up.pt)
- [Sérgio Estêvão](mailto:up201905680@edu.fe.up.pt)

[CPA @ M.EIC](https://sigarra.up.pt/feup/pt/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=486270)
