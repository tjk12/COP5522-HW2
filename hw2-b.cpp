/**
 * hw2-b.cpp
 *
 * A parallel C++ implementation of lower-triangular matrix-vector multiplication
 * using OpenMP. This version is optimized to avoid computations with zero
 * elements and uses modern C++ features.
 *
 * The workload for each row is unbalanced, making the choice of OpenMP scheduling
 * strategy critical. This program allows specifying the schedule via a command
 * line argument.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <string>
#include <omp.h>

// Helper function to get current time in microseconds
double microtime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// Main logic for lower-triangular matrix-vector multiplication
void mat_vec_mult_triangular(int n, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    // This pragma parallelizes the outer loop. The schedule is determined at runtime
    // based on the call to omp_set_schedule() in main(), which allows us to benchmark
    // different strategies without recompiling.
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; ++i) {
        // A local variable for accumulation. Each thread gets its own 'sum'.
        // This is crucial for correctness and avoids false sharing on the output vector C.
        float sum = 0.0f;
        // The inner loop only goes up to 'i', avoiding multiplication by zeros.
        for (int j = 0; j <= i; ++j) {
            sum += A[i * n + j] * B[j];
        }
        C[i] = sum; // Single write to the shared output vector C. No race condition.
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_n> [schedule]" << std::endl;
        std::cerr << "  [schedule] is optional (static, dynamic, guided) and defaults to 'guided'." << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    std::string schedule_type = (argc == 3) ? argv[2] : "guided"; // Default to guided

    if (n <= 0) {
        std::cerr << "Error: Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    // Set the OpenMP schedule type based on the command-line argument
    if (schedule_type == "static") {
        omp_set_schedule(omp_sched_static, 0);
    } else if (schedule_type == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, 1); // Use chunk size 1 for fine-grained dynamic
    } else if (schedule_type == "guided") {
        omp_set_schedule(omp_sched_guided, 0);
    } else {
        std::cerr << "Error: Invalid schedule type '" << schedule_type << "'." << std::endl;
        return 1;
    }

    // Allocate matrices
    std::vector<float> A(n * n, 0.0f), B(n), C(n);

    // Initialize as a lower triangular matrix
    for (int i = 0; i < n; ++i) {
        B[i] = 1.0f / (i + 2.0f);
        for (int j = 0; j <= i; ++j) {
            A[i * n + j] = 1.0f / (i + j + 2.0f);
        }
    }

    // Warm-up run
    mat_vec_mult_triangular(n, A, B, C);

    // Timed run
    double time1 = microtime();
    mat_vec_mult_triangular(n, A, B, C);
    double time2 = microtime();

    double elapsed_time_us = time2 - time1;
    double elapsed_time_sec = elapsed_time_us / 1e6;

    // Calculate performance in Gflop/s
    double gflops = 0.0;
    if (elapsed_time_sec > 0.0) {
        // Total flops for triangular matrix: sum of 2*i for i=1..n => n*(n+1)
        double total_flops = (double)n * (double)(n + 1);
        gflops = total_flops / (elapsed_time_sec * 1e9);
    }

    std::cout << "Execution Time: " << elapsed_time_us << " us" << std::endl;
    std::cout << "Matrix Size: " << n << "x" << n << ", Schedule: " << schedule_type << std::endl;
    std::cout << "Threads used: " << omp_get_max_threads() << std::endl;
    std::cout << "Performance (Gflop/s): " << gflops << std::endl;

    return 0;
}
