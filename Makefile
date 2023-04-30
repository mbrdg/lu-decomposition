# Makefile
# CPA @ M.EIC, 2023
# Authors: Miguel Rodrigues & Sérgio Estêvão
CXX=g++
CXXFLAGS=-std=c++20 -O3 -Wall -Wextra -Werror -pedantic -Wconversion -Wshadow

.PHONY: clean all

all: lu lublk luomp

%: src/%.cpp
	@mkdir -p bin/
	$(CXX) $(CXXFLAGS) $< -fopenmp -o bin/$@.out

clean:
	$(RM) bin/*.out
