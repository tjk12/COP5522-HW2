#!/bin/bash
# run_benchmarks.sh (v3)
# This script automates the entire benchmarking process for HW2.
# It compiles the code with different optimization profiles, runs a
# comprehensive set of tests for each, and saves all results to a
# structured JSON file for later analysis.

# --- Configuration ---
RESULTS_FILE="results.json"
MATRIX_SIZES_GENERAL=(1024 2048 4096 8192)
MATRIX_SIZE_STRONG=4096
MATRIX_SIZE_WEAK_BASE=2048
THREAD_COUNTS=(1 2 4 8 12 16)
SCHEDULES=("static" "dynamic" "guided")

# --- NEW: Define Optimization Profiles to Test ---
# Each key corresponds to a 'make' target (e.g., 'make all-O3')
declare -A OPTIMIZATION_PROFILES
OPTIMIZATION_PROFILES=(
    ["O3_default"]="-O3 -march=native -mavx2 -mfma"
    ["O2_optimized"]="-O2 -march=native -mavx2 -mfma"
    ["O3_unrolled"]="-O3 -march=native -mavx2 -mfma -funroll-loops"
)
MAKE_TARGETS=("all-O3" "all-O2" "all-unroll") # Corresponds to the keys above

# --- Helper Function ---
# Executes a command and extracts the Gflop/s value from its output
run_and_get_gflops() {
    local command_output
    command_output=$(eval "$1" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "error"
        return
    fi
    # Use grep and awk for more robust parsing of the floating point number
    echo "$command_output" | grep "Performance (Gflop/s):" | awk '{print $NF}'
}

# --- Main Script ---
echo "--- Starting Comprehensive Benchmark Process ---"

# Initialize JSON structure
echo "{" > "$RESULTS_FILE"
FIRST_PROFILE=true

# 1. Loop through each optimization profile
for i in "${!MAKE_TARGETS[@]}"; do
    PROFILE_KEY=$(echo "${!OPTIMIZATION_PROFILES[@]}" | cut -d' ' -f$((i+1)))
    MAKE_TARGET=${MAKE_TARGETS[$i]}

    echo -e "\n--------------------------------------------------"
    echo "--- BENCHMARKING PROFILE: $PROFILE_KEY ---"
    echo "--------------------------------------------------"

    # Compile the code with the current profile
    echo "Compiling with 'make $MAKE_TARGET'..."
    make clean > /dev/null && make "$MAKE_TARGET"
    if [ $? -ne 0 ]; then
        echo "!!! Compilation Failed for profile $PROFILE_KEY. Skipping. !!!"
        continue
    fi

    if [ "$FIRST_PROFILE" = false ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    FIRST_PROFILE=false

    # Start JSON object for this profile
    echo "\"$PROFILE_KEY\": {" >> "$RESULTS_FILE"
    echo "\"compiler_flags\": \"${OPTIMIZATION_PROFILES[$PROFILE_KEY]}\"," >> "$RESULTS_FILE"

    # --- 2. General Performance Sweep ---
    echo -e "\nRunning General Performance Sweep for '$PROFILE_KEY'..."
    echo "\"general_perf\": {" >> "$RESULTS_FILE"
    echo "\"hw2_a\": {" >> "$RESULTS_FILE"
    FIRST_SIZE=true
    for N in "${MATRIX_SIZES_GENERAL[@]}"; do
        if [ "$FIRST_SIZE" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        FIRST_SIZE=false
        echo "\"N$N\": {" >> "$RESULTS_FILE"
        FIRST_THREAD=true
        for T in "${THREAD_COUNTS[@]}"; do
            if [ "$FIRST_THREAD" = false ]; then echo "," >> "$RESULTS_FILE"; fi
            FIRST_THREAD=false
            GFLOPS=$(run_and_get_gflops "OMP_NUM_THREADS=$T ./hw2-a $N")
            echo "\"T$T\": \"$GFLOPS\"" >> "$RESULTS_FILE"
        done
        echo "}" >> "$RESULTS_FILE"
    done
    echo "}," >> "$RESULTS_FILE"
    echo "\"hw2_b\": {" >> "$RESULTS_FILE"
    FIRST_SIZE=true
    for N in "${MATRIX_SIZES_GENERAL[@]}"; do
        if [ "$FIRST_SIZE" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        FIRST_SIZE=false
        echo "\"N$N\": {" >> "$RESULTS_FILE"
        FIRST_SCHEDULE=true
        for S in "${SCHEDULES[@]}"; do
            if [ "$FIRST_SCHEDULE" = false ]; then echo "," >> "$RESULTS_FILE"; fi
            FIRST_SCHEDULE=false
            echo "\"schedule_$S\": {" >> "$RESULTS_FILE"
            FIRST_THREAD=true
            for T in "${THREAD_COUNTS[@]}"; do
                if [ "$FIRST_THREAD" = false ]; then echo "," >> "$RESULTS_FILE"; fi
                FIRST_THREAD=false
                GFLOPS=$(run_and_get_gflops "OMP_NUM_THREADS=$T ./hw2-b $N $S")
                echo "\"T$T\": \"$GFLOPS\"" >> "$RESULTS_FILE"
            done
            echo "}" >> "$RESULTS_FILE"
        done
        echo "}" >> "$RESULTS_FILE"
    done
    echo "}" >> "$RESULTS_FILE"
    echo "}," >> "$RESULTS_FILE" # End general_perf

    # --- 3. Strong Scaling Test ---
    echo -e "\nRunning Strong Scaling Test (N=$MATRIX_SIZE_STRONG) for '$PROFILE_KEY'..."
    echo "\"strong_scaling\": {" >> "$RESULTS_FILE"
    echo "\"hw2_a\": {" >> "$RESULTS_FILE"
    FIRST_THREAD=true
    for T in "${THREAD_COUNTS[@]}"; do
        if [ "$FIRST_THREAD" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        FIRST_THREAD=false
        GFLOPS=$(run_and_get_gflops "OMP_NUM_THREADS=$T ./hw2-a $MATRIX_SIZE_STRONG")
        echo "\"T$T\": \"$GFLOPS\"" >> "$RESULTS_FILE"
    done
    echo "}," >> "$RESULTS_FILE"
    echo "\"hw2_b_guided\": {" >> "$RESULTS_FILE"
    FIRST_THREAD=true
    for T in "${THREAD_COUNTS[@]}"; do
        if [ "$FIRST_THREAD" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        FIRST_THREAD=false
        GFLOPS=$(run_and_get_gflops "OMP_NUM_THREADS=$T ./hw2-b $MATRIX_SIZE_STRONG guided")
        echo "\"T$T\": \"$GFLOPS\"" >> "$RESULTS_FILE"
    done
    echo "}" >> "$RESULTS_FILE"
    echo "}," >> "$RESULTS_FILE" # End strong_scaling

    # --- 4. Weak Scaling Test ---
    echo -e "\nRunning Weak Scaling Test (Base N=$MATRIX_SIZE_WEAK_BASE) for '$PROFILE_KEY'..."
    echo "\"weak_scaling\": {" >> "$RESULTS_FILE"
    echo "\"hw2_a\": {" >> "$RESULTS_FILE"
    FIRST_THREAD=true
    for T in "${THREAD_COUNTS[@]}"; do
        if [ "$FIRST_THREAD" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        FIRST_THREAD=false
        N_WEAK=$(echo "scale=0; sqrt($T) * $MATRIX_SIZE_WEAK_BASE / 1" | bc)
        GFLOPS=$(run_and_get_gflops "OMP_NUM_THREADS=$T ./hw2-a $N_WEAK")
        echo "\"T$T\": {\"N\": \"$N_WEAK\", \"gflops\": \"$GFLOPS\"}" >> "$RESULTS_FILE"
    done
    echo "}," >> "$RESULTS_FILE"
    echo "\"hw2_b_guided\": {" >> "$RESULTS_FILE"
    FIRST_THREAD=true
    for T in "${THREAD_COUNTS[@]}"; do
        if [ "$FIRST_THREAD" = false ]; then echo "," >> "$RESULTS_FILE"; fi
        FIRST_THREAD=false
        N_WEAK=$(echo "scale=0; sqrt($T) * $MATRIX_SIZE_WEAK_BASE / 1" | bc)
        GFLOPS=$(run_and_get_gflops "OMP_NUM_THREADS=$T ./hw2-b $N_WEAK guided")
        echo "\"T$T\": {\"N\": \"$N_WEAK\", \"gflops\": \"$GFLOPS\"}" >> "$RESULTS_FILE"
    done
    echo "}" >> "$RESULTS_FILE"
    echo "}" >> "$RESULTS_FILE" # End weak_scaling

    echo "}" >> "$RESULTS_FILE" # End profile object

done

echo "}" >> "$RESULTS_FILE" # End main JSON object

echo -e "\n--------------------------------------------------"
echo "--- Comprehensive Benchmarking Complete ---"
echo "All results have been saved to $RESULTS_FILE"
echo -e "\n--- Automatically Generating PDF Report ---"
python report.py

echo -e "\n--- Process Finished ---"

