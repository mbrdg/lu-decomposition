# Makefile
# CPA @ M.EIC, 2023
# Authors: Miguel Rodrigues & Sérgio Estêvão
CXX=g++
CXXFLAGS=-std=c++20 -O3 -Wall -Wextra -Werror -pedantic -Wconversion -Wshadow

CUXX=nvcc	#/usr/local/cuda-12/bin/nvcc
CUDAFLAGS=-O3 --expt-relaxed-constexpr

SYCLXX=clang++
SYCLFLAGS=-std=c++20 -O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
	-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda \
	--cuda-gpu-arch=sm_75

.PHONY: clean all

all: lu lublk luomp lucuda lusycl

%: src/%.cpp
	@mkdir -p bin/
	$(CXX) $(CXXFLAGS) $< -fopenmp -o bin/$@.out

%: src/%.cu
	@mkdir -p bin/
	$(CUXX) $(CUDAFLAGS) $< -o bin/$@.out

## TODO: make it compile to OpenMP or in the CPU
lusycl: src/lusycl.cpp
	@mkdir -p bin/
	$(SYCLXX) $(SYCLFLAGS) $< -o bin/$@.out

clean:
	$(RM) bin/*.out
