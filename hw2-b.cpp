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
#include <cmath>
#include <algorithm>

using namespace std;

// Forward declaration
void Mv_mult_openmp(int n, const vector<double>& A, const vector<double>& B, vector<double>& C, const string& schedule_type);

// --- Timing and Utility Functions ---
double microtime() {
    return chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

void print_usage(const char* prog_name) {
    cerr << "Usage: " << prog_name << " <matrix_size_n> [schedule_type]" << endl;
    cerr << "  <schedule_type> is optional. Options: static, dynamic, guided. Defaults to guided." << endl;
}

// --- Main Logic ---
int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        print_usage(argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        cerr << "Error: Matrix size must be a positive integer." << endl;
        return 1;
    }

    string schedule_type = "guided"; // Default schedule
    if (argc == 3) {
        schedule_type = argv[2];
        if (schedule_type != "static" && schedule_type != "dynamic" && schedule_type != "guided") {
            cerr << "Error: Invalid schedule type '" << schedule_type << "'." << endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    vector<double> A(n * n), B(n), C(n);

    // Initialize as a lower-triangular matrix
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        B[i] = 1.0 / (i + 2.0);
        for (int j = 0; j < n; ++j) {
            if (j <= i) {
                A[i * n + j] = 1.0 / (i + j + 2.0);
            } else {
                A[i * n + j] = 0.0;
            }
        }
    }

    double time1 = microtime();
    Mv_mult_openmp(n, A, B, C, schedule_type);
    double time2 = microtime();

    double elapsed_us = time2 - time1;
    double gflops = (2.0 * n * (n + 1.0) / 2.0) / (elapsed_us * 1e3);

    cout << "Threads: " << omp_get_max_threads() << ", N: " << n 
         << ", Schedule: " << schedule_type
         << ", Time: " << elapsed_us / 1e6 << " s"
         << ", Performance: " << gflops << " Gflop/s" << endl;

    return 0;
}

void Mv_mult_openmp(int n, const vector<double>& A, const vector<double>& B, vector<double>& C, const string& schedule_type) {
    // Set the schedule based on the input string
    if (schedule_type == "static") {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j <= i; ++j) { // Important: only loop to i
                sum += A[i * n + j] * B[j];
            }
            C[i] = sum;
        }
    } else if (schedule_type == "dynamic") {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j <= i; ++j) {
                sum += A[i * n + j] * B[j];
            }
            C[i] = sum;
        }
    } else { // default to guided
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j <= i; ++j) {
                sum += A[i * n + j] * B[j];
            }
            C[i] = sum;
        }
    }
}

