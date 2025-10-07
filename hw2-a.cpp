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

using namespace std;

// Helper function to get current time in microseconds
double microtime() {
    auto now = chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return chrono::duration_cast<chrono::microseconds>(duration).count();
}

// Main logic for dense matrix-vector multiplication
void mat_vec_mult(int n, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    // Initialize C to zeros
    fill(C.begin(), C.end(), 0.0f);

    // Using i-k loop order for cache-friendly sequential access to A and B.
    // This is the key change to improve performance significantly.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            C[i] += A[i * n + k] * B[k];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size_n>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        cerr << "Error: Matrix size must be a positive integer." << endl;
        return 1;
    }

    // Allocate and initialize matrices
    vector<float> A(n * n), B(n), C(n);
    for (int i = 0; i < n; ++i) {
        B[i] = 1.0f / (i + 2.0f);
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = 1.0f / (i + j + 2.0f);
        }
    }

    // Warm-up run to stabilize CPU frequency and cache
    mat_vec_mult(n, A, B, C);

    // Timed run
    double time1 = microtime();
    mat_vec_mult(n, A, B, C);
    double time2 = microtime();

    double elapsed_time_us = time2 - time1;
    double elapsed_time_sec = elapsed_time_us / 1e6;

    // Calculate performance in Gflop/s
    double gflops = 0.0;
    if (elapsed_time_sec > 0.0) {
        double total_flops = 2.0 * n * n; // n*n additions and n*n multiplications
        gflops = total_flops / (elapsed_time_sec * 1e9);
    }
    
    cout << "Execution Time: " << elapsed_time_us << " us" << endl;
    cout << "Matrix Size: " << n << "x" << n << endl;
    cout << "Threads used: " << omp_get_max_threads() << endl;
    cout << "Performance (Gflop/s): " << gflops << endl;

    return 0;
}