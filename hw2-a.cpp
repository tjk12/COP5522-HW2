/**
 * hw2-a.cpp
 *
 * A parallel C++ implementation of dense matrix-vector multiplication using OpenMP.
 *
 * This version uses modern C++ features like std::vector for memory management
 * and std::chrono for high-resolution timing, while retaining the OpenMP
 * parallelization strategy for the main computation loop.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <string>
#include <cstring>
#include <omp.h>

// --- Helper Functions ---
double microtime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

void print_usage(const char* prog_name) {
    std::cerr << "USAGE: " << prog_name << " <Matrix-Dimension>" << std::endl;
}

// --- Matrix-Vector Multiplication (Dense) ---
void MatVecMult(int n, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    // Initialize C to all zeros
    std::fill(C.begin(), C.end(), 0.0f);

    // Parallelize the outer loop using OpenMP.
    // The k-i loop order is preserved from your Mv.cpp for cache efficiency.
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            C[i] += A[i * n + k] * B[k];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: Matrix dimension must be a positive integer." << std::endl;
        return 1;
    }

    std::vector<float> A(n * n);
    std::vector<float> B(n);
    std::vector<float> C(n);

    // Initialize matrices A and B (vector)
    for (int i = 0; i < n; i++) {
        B[i] = 1.0f / (i + 2.0f);
        for (int j = 0; j < n; j++) {
            A[i * n + j] = 1.0f / (i + j + 2.0f);
        }
    }

    // Warm-up run to stabilize CPU frequency and cache
    MatVecMult(n, A, B, C);

    double time1 = microtime();
    MatVecMult(n, A, B, C);
    double time2 = microtime();

    double elapsed_us = time2 - time1;
    double gflops = (2.0 * n * n) / (elapsed_us * 1e3);

    std::cout << "Matrix Size: " << n << "x" << n << std::endl;
    std::cout << "Threads used: " << omp_get_max_threads() << std::endl;
    std::cout << "Time: " << elapsed_us << " us" << std::endl;
    std::cout << "Performance (Gflop/s): " << gflops << std::endl;

    return 0;
}