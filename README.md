# LU Decomposition

LU decomposition for solving linear systems using SYCL and CUDA.

## Usage

The usage is really dependent on the platform that is being used.
Furthermore, you are required to have a NVidia GPU to execute this project
as we made use of CUDA itself and SYCL with the CUDA backend.

## Dependencies

For this project we use the `nvcc` the NVidia CUDA compiler
and the `dpcpp` the Data Parallel C++ compiler which implements the SYCL specification.

Also, when using the latter we use an extension to provide a CUDA backend.
The extension is available for download
[here](https://developer.codeplay.com/products/oneapi/nvidia/2023.1.0/guides/get-started-guide-nvidia).

It is up to you to figure out how to install those and make them run in your system.

## Authors

- [Miguel Rodrigues](mailto:up201906042@edu.fe.up.pt)
- [Sérgio Estêvão](mailto:up201905680@edu.fe.up.pt)

[CPA @ M.EIC](https://sigarra.up.pt/feup/pt/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=486270)

