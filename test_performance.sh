#!/bin/bash

# This script compiles and runs C++ benchmarks, tests multiple optimization
# profiles, and generates a comprehensive performance report.

# --- Dependency Check ---
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' (a command-line calculator) is not installed." >&2
    exit 1
fi

# --- Internal Configuration ---
RESULTS_FILE="results.json"
# Matrix sizes for general performance and strong scaling sweeps
SIZES=(1024 2048 4096 8192)
# Thread counts to test
THREADS=(1 2 4 8 12 16)
# Base matrix sizes for the DIFFERENT weak scaling experiments
WEAK_SCALING_BASE_SIZES=(1024 2048)

# Define compiler optimization profiles to test
# Format: "key:make_target:human_readable_flags"
OPTIMIZATION_PROFILES=(
    "O3_default:all-O3:-O3 -march=native -mavx2 -mfma"
    "O2_optimized:all-O2:-O2 -march=native -mavx2 -mfma"
    "O3_unrolled:all-unroll:-O3 -march=native -mavx2 -mfma -funroll-loops"
)

# --- Helper Functions ---
# Runs a command, captures the 'Performance' line, and extracts the Gflop/s value
run_and_get_gflops() {
    local command_output
    command_output=$(eval "$@" 2>&1)
    if [ $? -eq 0 ]; then
        echo "$command_output" | grep "Performance" | awk '{print $NF}'
    else
        echo "" # Return empty on error
    fi
}

# --- Main Script ---
echo "--- Starting Comprehensive Benchmark Process ---"

# Initialize JSON file
echo "{" > "$RESULTS_FILE"

first_profile=true
for profile_info in "${OPTIMIZATION_PROFILES[@]}"; do
    # Parse profile info
    IFS=':' read -r profile_key make_target flags <<< "$profile_info"

    echo ""
    echo "--------------------------------------------------"
    echo "--- BENCHMARKING PROFILE: $profile_key ---"
    echo "--------------------------------------------------"

    # Compile the code with the current profile
    echo "Compiling with 'make $make_target'..."
    make clean > /dev/null && make "$make_target"
    if [ $? -ne 0 ]; then
        echo "!!! Compilation Failed for profile $profile_key. Skipping. !!!"
        continue
    fi

    # Add comma separator for JSON entries
    if [ "$first_profile" = false ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    first_profile=false

    # --- Start JSON entry for this profile ---
    echo "  \"$profile_key\": {" >> "$RESULTS_FILE"
    echo "    \"compiler_flags\": \"$flags\"," >> "$RESULTS_FILE"
    echo "    \"general_perf\": {" >> "$RESULTS_FILE"
    echo "      \"hw2_a\": {" >> "$RESULTS_FILE"

    # 1. General Performance Sweep (Strong Scaling Data Source)
    echo ""
    echo "Running General Performance Sweep for '$profile_key'..."
    first_size=true
    for N in "${SIZES[@]}"; do
        if [ "$first_size" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        first_size=false
        echo "      \"N$N\": {" >> "$RESULTS_FILE"
        first_thread=true
        for T in "${THREADS[@]}"; do
            gflops=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-a "$N")
            if [ "$first_thread" = false ]; then echo "," >> "$RESULTS_FILE"; fi
            echo "        \"T$T\": \"$gflops\"" >> "$RESULTS_FILE"
            first_thread=false
        done
        echo "      }" >> "$RESULTS_FILE"
    done
    echo "      }," >> "$RESULTS_FILE" # end hw2_a
    echo "      \"hw2_b\": {" >> "$RESULTS_FILE"
    
    # Test hw2-b with all schedules
    SCHEDULES=("static" "dynamic" "guided")
    first_size=true
    for N in "${SIZES[@]}"; do
        if [ "$first_size" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        first_size=false
        echo "      \"N$N\": {" >> "$RESULTS_FILE"
        first_sched=true
        for SCHED in "${SCHEDULES[@]}"; do
            if [ "$first_sched" = false ]; then echo "," >> "$RESULTS_FILE"; fi
            first_sched=false
            echo "        \"schedule_$SCHED\": {" >> "$RESULTS_FILE"
            first_thread=true
            for T in "${THREADS[@]}"; do
                gflops=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-b "$N" "$SCHED")
                if [ "$first_thread" = false ]; then echo "," >> "$RESULTS_FILE"; fi
                echo "          \"T$T\": \"$gflops\"" >> "$RESULTS_FILE"
                first_thread=false
            done
            echo "        }" >> "$RESULTS_FILE"
        done
        echo "      }" >> "$RESULTS_FILE"
    done
    echo "      }" >> "$RESULTS_FILE" # end hw2_b
    echo "    }," >> "$RESULTS_FILE" # end general_perf

    # 2. Dedicated Weak Scaling Tests
    echo ""
    echo "Running Weak Scaling Tests for '$profile_key'..."
    echo "    \"weak_scaling\": {" >> "$RESULTS_FILE"
    first_base_size=true
    for BASE_N in "${WEAK_SCALING_BASE_SIZES[@]}"; do
        if [ "$first_base_size" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        first_base_size=false
        echo "      \"N${BASE_N}\": {" >> "$RESULTS_FILE"
        echo "        \"hw2_a\": {" >> "$RESULTS_FILE"
        first_thread=true
        for T in "${THREADS[@]}"; do
            N_weak=$(echo "sqrt($T) * $BASE_N" | bc -l | awk '{print int($1+0.5)}')
            gflops=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-a "$N_weak")
            if [ "$first_thread" = false ]; then echo "," >> "$RESULTS_FILE"; fi
            echo "          \"T$T\": {\"N\": \"$N_weak\", \"gflops\": \"$gflops\"}" >> "$RESULTS_FILE"
            first_thread=false
        done
        echo "        }," >> "$RESULTS_FILE" # end hw2_a
        echo "        \"hw2_b_guided\": {" >> "$RESULTS_FILE"
        first_thread=true
        for T in "${THREADS[@]}"; do
            N_weak=$(echo "sqrt($T * ($BASE_N^2 + $BASE_N) / 2) * sqrt(2)" | bc -l | awk '{print int($1+0.5)}')
            gflops=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-b "$N_weak" "guided")
            if [ "$first_thread" = false ]; then echo "," >> "$RESULTS_FILE"; fi
            echo "          \"T$T\": {\"N\": \"$N_weak\", \"gflops\": \"$gflops\"}" >> "$RESULTS_FILE"
            first_thread=false
        done
        echo "        }" >> "$RESULTS_FILE" # end hw2_b
        echo "      }" >> "$RESULTS_FILE"
    done
    echo "    }" >> "$RESULTS_FILE" # end weak_scaling
    echo "  }" >> "$RESULTS_FILE" # end profile
done

# Finalize JSON file
echo "}" >> "$RESULTS_FILE"

echo ""
echo "--------------------------------------------------"
echo "--- Comprehensive Benchmarking Complete ---"
echo "All results have been saved to $RESULTS_FILE"
echo ""

# 3. Automatically generate the final report
echo "--- Automatically Generating PDF Report ---"
python3 report.py
if [ $? -ne 0 ]; then
    echo "!!! Report Generation Failed. Please check create_report.py and its dependencies. !!!"
    exit 1
fi
echo "--- Process Finished ---"

