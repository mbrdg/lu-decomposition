# Makefile
# CPA @ M.EIC, 2023
# Authors: Miguel Rodrigues & Sérgio Estêvão
CXX=g++
CXXFLAGS=-std=c++20 -O3 -Wall -Wextra -Werror -pedantic -Wconversion -Wshadow

CUXX=nvcc
CUDAFLAGS=--expt-relaxed-constexpr

.PHONY: clean all

all: lu lublk luomp lucuda

%: src/%.cpp
	@mkdir -p bin/
	$(CXX) $(CXXFLAGS) $< -fopenmp -o bin/$@.out

%: src/%.cu
	@mkdir -p bin/
	$(CUXX) $(CUDAFLAGS) $< -o bin/$@.out


clean:
	$(RM) bin/*.out
