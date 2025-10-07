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
#include <cstring>
#include <omp.h>

using namespace std;

// Helper function to get current time in microseconds
double microtime() {
    auto now = chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return chrono::duration_cast<chrono::microseconds>(duration).count();
}

void mat_vec_mult_triangular(int n, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    // A temporary vector for each thread to accumulate results, preventing false sharing
    vector<float> C_local(C.size(), 0.0f);

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            C_local[i] += A[i * n + j] * B[j];
        }
    }

    // Reduction: combine the results from local copies into the final vector C
    // This part is sequential but fast
    fill(C.begin(), C.end(), 0.0f);
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = C_local[i];
    }
}


int main(int argc, char **argv) {
    int n;
    string schedule_type;

    // Handle command line arguments to allow for a default schedule
    if (argc == 2) {
        n = atoi(argv[1]);
        schedule_type = "static"; // Default schedule if only size is provided
    } else if (argc == 3) {
        n = atoi(argv[1]);
        schedule_type = argv[2];
    } else {
        cerr << "Usage: " << argv[0] << " <matrix_size_n> [schedule]" << endl;
        cerr << "  [schedule] is optional and defaults to 'static'." << endl;
        cerr << "  Available schedules: static, dynamic, guided" << endl;
        return 1;
    }
    
    if (n <= 0) {
        cerr << "Error: Matrix size must be a positive integer." << endl;
        return 1;
    }

    // Set the OMP schedule type from the command line argument
    if (schedule_type == "static") {
        omp_set_schedule(omp_sched_static, 0);
    } else if (schedule_type == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, 0);
    } else if (schedule_type == "guided") {
        omp_set_schedule(omp_sched_guided, 0);
    } else {
        cerr << "Error: Invalid schedule type '" << schedule_type << "'." << endl;
        return 1;
    }

    // Allocate matrices
    vector<float> A(n * n, 0.0f), B(n), C(n);

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
    // FIX: Add a safety check to prevent division by zero
    if (elapsed_time_sec > 0.0) {
        // Total flops for triangular matrix is n*(n+1)
        double total_flops = (double)n * (n + 1);
        gflops = total_flops / (elapsed_time_sec * 1e9);
    }

    cout << "Execution Time: " << elapsed_time_us << " us" << endl;
    cout << "Matrix Size: " << n << "x" << n << ", Schedule: " << schedule_type << endl;
    cout << "Threads used: " << omp_get_max_threads() << endl;
    cout << "Performance (Gflop/s): " << gflops << endl;

    return 0;
}