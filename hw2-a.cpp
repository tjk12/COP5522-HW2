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
#include <omp.h>

// Helper function to get current time in microseconds for precise timing
double microtime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// Main logic for dense matrix-vector multiplication
void mat_vec_mult(int n, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    // The outer loop is parallelized. Each thread handles a distinct set of rows ('i').
    // 'static' scheduling is efficient here because the workload for each row is the same.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f; // Use a local variable to prevent race conditions.
        for (int k = 0; k < n; ++k) {
            sum += A[i * n + k] * B[k];
        }
        C[i] = sum; // Each thread writes to a unique C[i], so no conflict occurs.
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_n>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    // Allocate and initialize matrices using std::vector
    std::vector<float> A(n * n), B(n), C(n);
    for (int i = 0; i < n; ++i) {
        B[i] = 1.0f / (i + 2.0f);
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = 1.0f / (i + j + 2.0f);
        }
    }

    // Warm-up run to stabilize CPU frequency and ensure caches are loaded
    mat_vec_mult(n, A, B, C);

    // Timed run for performance measurement
    double time1 = microtime();
    mat_vec_mult(n, A, B, C);
    double time2 = microtime();

    double elapsed_time_us = time2 - time1;
    double elapsed_time_sec = elapsed_time_us / 1e6;

    // Calculate performance in Gflop/s (Billion Floating Point Operations Per Second)
    double gflops = 0.0;
    if (elapsed_time_sec > 0.0) {
        // Total operations: n*n multiplications and n*n additions = 2*n^2 flops
        double total_flops = 2.0 * (double)n * (double)n;
        gflops = total_flops / (elapsed_time_sec * 1e9);
    }

    std::cout << "Execution Time: " << elapsed_time_us << " us" << std::endl;
    std::cout << "Matrix Size: " << n << "x" << n << std::endl;
    std::cout << "Threads used: " << omp_get_max_threads() << std::endl;
    std::cout << "Performance (Gflop/s): " << gflops << std::endl;

    return 0;
}
