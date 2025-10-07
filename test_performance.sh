#!/bin/bash
# Comprehensive benchmarking script for HW2

# --- Configuration ---
RESULTS_FILE="results.json"
# Thread counts for general and strong scaling tests
THREADS=(1 2 4 8 12 16) 
# Matrix sizes for general performance table
SIZES=(1024 2048 4096 8192)
# OpenMP schedules to test for hw2-b
SCHEDULES=("static" "dynamic" "guided")
# Fixed large size for strong scaling test
STRONG_SCALING_N=4096
# Base size for weak scaling (work per thread)
WEAK_SCALING_BASE_N=2048

# --- Dependency Check ---
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. It's needed to build the JSON result file." >&2
    echo "On macOS: brew install jq" >&2
    echo "On Debian/Ubuntu: sudo apt-get install jq" >&2
    exit 1
fi
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' is not installed. It is needed for weak scaling calculations." >&2
    exit 1
fi

# --- Main Script ---
echo "--- Starting HW2 Benchmark Process ---"

# 1. Compile all C++ code
echo "Compiling C++ executables via make..."
make all
if [ $? -ne 0 ]; then
    echo "!!! Compilation Failed. Aborting. !!!"
    exit 1
fi
echo "Compilation successful."

# Initialize JSON file
echo "{}" > $RESULTS_FILE

# --- Helper function to run a command and parse Gflop/s ---
run_and_get_gflops() {
    # stderr is redirected to stdout to capture potential errors
    output=$(OMP_NUM_THREADS=$1 $2 $3 $4 2>&1)
    
    # ROBUST PARSING FIX: Instead of relying on word count (awk), specifically extract
    # the last floating-point number from the line containing "Performance".
    # This is more resilient to minor differences in the C++ cout statements.
    gflops=$(echo "$output" | grep "Performance" | grep -o '[0-9]\+\.\?[0-9]*' | tail -n 1)
    
    if [[ -z "$gflops" || "$output" == *"Error"* ]]; then
        echo "error"
    else
        echo "$gflops"
    fi
}

# 2. General Performance Table Data
echo -e "\n--- Running General Benchmarks for Performance Table ---"
for n in "${SIZES[@]}"; do
    for t in "${THREADS[@]}"; do
        # Run hw2-a
        gflops_a=$(run_and_get_gflops $t ./hw2-a $n)
        echo "hw2-a: N=$n, Threads=$t -> $gflops_a Gflop/s"
        jq ".general_perf.hw2_a.N$n.T$t = \"$gflops_a\"" $RESULTS_FILE > tmp.$$.json && mv tmp.$$.json $RESULTS_FILE

        # Run hw2-b for all schedules
        for sched in "${SCHEDULES[@]}"; do
            gflops_b=$(run_and_get_gflops $t ./hw2-b $n $sched)
            echo "hw2-b: N=$n, Threads=$t, Schedule=$sched -> $gflops_b Gflop/s"
            jq ".general_perf.hw2_b.N$n.schedule_$sched.T$t = \"$gflops_b\"" $RESULTS_FILE > tmp.$$.json && mv tmp.$$.json $RESULTS_FILE
        done
    done
done


# 3. Strong Scaling Data
echo -e "\n--- Running Strong Scaling Benchmark (N = $STRONG_SCALING_N) ---"
for t in "${THREADS[@]}"; do
    # Run hw2-a
    gflops_a=$(run_and_get_gflops $t ./hw2-a $STRONG_SCALING_N)
    echo "hw2-a: Strong scaling, Threads=$t -> $gflops_a Gflop/s"
    jq ".strong_scaling.hw2_a.T$t = \"$gflops_a\"" $RESULTS_FILE > tmp.$$.json && mv tmp.$$.json $RESULTS_FILE
    
    # Run hw2-b (best schedule, assumed 'guided' for this test)
    gflops_b=$(run_and_get_gflops $t ./hw2-b $STRONG_SCALING_N "guided")
    echo "hw2-b (guided): Strong scaling, Threads=$t -> $gflops_b Gflop/s"
    jq ".strong_scaling.hw2_b_guided.T$t = \"$gflops_b\"" $RESULTS_FILE > tmp.$$.json && mv tmp.$$.json $RESULTS_FILE
done

# 4. Weak Scaling Data
echo -e "\n--- Running Weak Scaling Benchmark (Base N = $WEAK_SCALING_BASE_N) ---"
for t in "${THREADS[@]}"; do
    # Scale N to keep work per thread constant. Work ~ N^2, so N should scale with sqrt(threads)
    n_weak=$(echo "sqrt($t) * $WEAK_SCALING_BASE_N" | bc -l | awk '{printf "%d", $1}')
    
    # Run hw2-a
    gflops_a=$(run_and_get_gflops $t ./hw2-a $n_weak)
    echo "hw2-a: Weak scaling, Threads=$t, N=$n_weak -> $gflops_a Gflop/s"
    jq ".weak_scaling.hw2_a.T$t = {\"N\": \"$n_weak\", \"gflops\": \"$gflops_a\"}" $RESULTS_FILE > tmp.$$.json && mv tmp.$$.json $RESULTS_FILE

    # Run hw2-b (best schedule, assumed 'guided')
    gflops_b=$(run_and_get_gflops $t ./hw2-b $n_weak "guided")
    echo "hw2-b (guided): Weak scaling, Threads=$t, N=$n_weak -> $gflops_b Gflop/s"
    jq ".weak_scaling.hw2_b_guided.T$t = {\"N\": \"$n_weak\", \"gflops\": \"$gflops_b\"}" $RESULTS_FILE > tmp.$$.json && mv tmp.$$.json $RESULTS_FILE
done

echo -e "\n--- Benchmarking Complete! ---"
echo "Results saved to $RESULTS_FILE"

