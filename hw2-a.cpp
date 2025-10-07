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
#include <cstdlib> // For atoi

// Utility function to get current time in seconds using C++ chrono
double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    // Using duration<double> to get seconds with fractional part
    return std::chrono::duration<double>(duration).count();
}

/**
 * @brief Initializes a dense matrix A and a vector B.
 * @param A The matrix (passed by reference).
 * @param B The vector (passed by reference).
 * @param n The dimension of the matrix and vector.
 */
void init_data(std::vector<float>& A, std::vector<float>& B, int n) {
    for (int i = 0; i < n; i++) {
        B[i] = 1.0f / (i + 2.0f);
        for (int j = 0; j < n; j++) {
            A[i * n + j] = 1.0f / (i + j + 2.0f);
        }
    }
}

/**
 * @brief Performs dense matrix-vector multiplication C = A * B.
 * The outer loop over rows 'i' is parallelized with OpenMP.
 * @param A The input matrix.
 * @param B The input vector.
 * @param C The output vector.
 * @param n The dimension of the matrix and vector.
 */
void mat_vec_mult(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n) {
    // Parallelize the loop over rows. Each thread handles a distinct set of rows,
    // so there are no race conditions when writing to the output vector C.
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        // This inner loop is a dot product for the i-th row of A and vector B.
        // It should be efficiently vectorized by the compiler.
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * B[j];
        }
        C[i] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_dimension>" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: Matrix dimension must be a positive integer." << std::endl;
        return 1;
    }

    // Allocate memory using std::vector. It handles allocation and deallocation automatically.
    std::vector<float> A(n * n);
    std::vector<float> B(n);
    std::vector<float> C(n);

    // Initialize data
    init_data(A, B, n);

    // Warm-up run to stabilize system performance
    mat_vec_mult(A, B, C, n);

    // Timed run
    double start_time = get_time();
    mat_vec_mult(A, B, C, n);
    double end_time = get_time();

    double elapsed_time = end_time - start_time;
    
    // Calculate Gflop/s
    // For a dense matrix, flops = 2 * n * n (n multiplications and n additions per row)
    long long flops = 2LL * n * n;
    double gflops = (double)flops / (elapsed_time * 1e9);

    // std::cout is the C++ equivalent of printf
    std::cout << "Matrix Size: " << n << std::endl;
    std::cout << "Elapsed Time: " << std::fixed << elapsed_time << " s" << std::endl;
    std::cout << "Performance (Gflop/s): " << std::fixed << gflops << std::endl;

    return 0;
}
