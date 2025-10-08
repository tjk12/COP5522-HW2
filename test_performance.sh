#!/bin/bash

# run_benchmarks.sh (v3)
# This script automates the entire process of testing and report generation.
# CORRECTED: The logic for printing commas between JSON sections has been fixed
# to prevent trailing commas, which create an invalid JSON file.

echo "--- Starting Comprehensive Benchmark Process ---"

# --- Dependency Check ---
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' is not installed. Please install it to continue." >&2
    exit 1
fi

# --- Configuration ---
MATRIX_SIZES=(1024 2048 4096)
THREAD_COUNTS=(1 2 4 8 16 32)
SCHEDULES=("static" "dynamic" "guided")
OUTPUT_FILE="results.json"
COMPILER_PROFILES=(
    "O3_default:all-O3"
    "O2_optimized:all-O2"
    "O3_unrolled:all-unroll"
)

# --- Helper Function ---
run_and_get_gflops() {
    local command_output=$(eval "$@" 2>&1)
    if [ $? -eq 0 ]; then
        echo "$command_output" | grep "Performance" | awk '{print $3}'
    else
        echo ""
    fi
}

# --- Main Logic ---

echo "{" > "$OUTPUT_FILE"

first_profile=true
for profile_info in "${COMPILER_PROFILES[@]}"; do
    IFS=':' read -r profile_key make_target <<< "$profile_info"

    echo -e "\n--- Compiling with profile: $profile_key ($make_target) ---"
    make clean > /dev/null && make "$make_target"
    if [ $? -ne 0 ]; then
        echo "Error: make failed for target $make_target. Skipping profile."
        continue
    fi

    if [ "$first_profile" = false ]; then echo "," >> "$OUTPUT_FILE"; fi
    first_profile=false

    echo "  \"$profile_key\": {" >> "$OUTPUT_FILE"
    echo "    \"general_perf\": {" >> "$OUTPUT_FILE"
    
    # --- Strong Scaling for hw2-a ---
    echo "      \"hw2_a\": {" >> "$OUTPUT_FILE"
    first_size=true
    for N in "${MATRIX_SIZES[@]}"; do
        if [ "$first_size" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_size=false
        echo "        \"N$N\": {" >> "$OUTPUT_FILE"
        first_thread=true
        for T in "${THREAD_COUNTS[@]}"; do
            echo "Running hw2-a: N=$N, T=$T"
            GFLOPS=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-a "$N")
            if [ "$first_thread" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_thread=false
            echo "          \"T$T\": \"$GFLOPS\"" >> "$OUTPUT_FILE"
        done
        echo "        }" >> "$OUTPUT_FILE"
    done
    # Comma after hw2_a is correct because hw2_b follows.
    echo "      }," >> "$OUTPUT_FILE"

    # --- Strong Scaling for hw2-b ---
    echo "      \"hw2_b\": {" >> "$OUTPUT_FILE"
    first_size=true
    for N in "${MATRIX_SIZES[@]}"; do
        if [ "$first_size" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_size=false
        echo "        \"N$N\": {" >> "$OUTPUT_FILE"
        first_sched=true
        for S in "${SCHEDULES[@]}"; do
            if [ "$first_sched" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_sched=false
            echo "          \"schedule_$S\": {" >> "$OUTPUT_FILE"
            first_thread=true
            for T in "${THREAD_COUNTS[@]}"; do
                echo "Running hw2-b: N=$N, T=$T, S=$S"
                GFLOPS=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-b "$N" "$S")
                if [ "$first_thread" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_thread=false
                echo "            \"T$T\": \"$GFLOPS\"" >> "$OUTPUT_FILE"
            done
            echo "          }" >> "$OUTPUT_FILE"
        done
        echo "        }" >> "$OUTPUT_FILE"
    done
    # CORRECTED: No comma after hw2_b, as it's the last item in 'general_perf'.
    echo "      }" >> "$OUTPUT_FILE"
    # Comma after general_perf is correct because weak_scaling follows.
    echo "    }," >> "$OUTPUT_FILE"

    # --- Weak Scaling ---
    echo "    \"weak_scaling\": {" >> "$OUTPUT_FILE"
    first_size=true
    for BASE_N in "${MATRIX_SIZES[@]}"; do
        if [ "$first_size" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_size=false
        echo "      \"N$BASE_N\": {" >> "$OUTPUT_FILE"
        
        echo "        \"hw2_a\": {" >> "$OUTPUT_FILE"
        first_thread=true
        for T in "${THREAD_COUNTS[@]}"; do
            WEAK_N=$(echo "sqrt($T) * $BASE_N" | bc -l | awk '{printf "%d\n", $1}')
            echo "Running weak scaling hw2-a: BaseN=$BASE_N, T=$T -> WeakN=$WEAK_N"
            GFLOPS=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-a "$WEAK_N")
            if [ "$first_thread" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_thread=false
            echo "          \"T$T\": { \"n\": \"$WEAK_N\", \"gflops\": \"$GFLOPS\" }" >> "$OUTPUT_FILE"
        done
        echo "        }," >> "$OUTPUT_FILE" # Comma after hw2_a is correct
        
        echo "        \"hw2_b_guided\": {" >> "$OUTPUT_FILE"
        first_thread=true
        for T in "${THREAD_COUNTS[@]}"; do
            WEAK_N=$(echo "sqrt($T * ($BASE_N^2 + $BASE_N))" | bc -l | awk '{printf "%d\n", $1}')
            echo "Running weak scaling hw2-b: BaseN=$BASE_N, T=$T -> WeakN=$WEAK_N"
            GFLOPS=$(OMP_NUM_THREADS=$T run_and_get_gflops ./hw2-b "$WEAK_N" "guided")
            if [ "$first_thread" = false ]; then echo "," >> "$OUTPUT_FILE"; fi; first_thread=false
            echo "          \"T$T\": { \"n\": \"$WEAK_N\", \"gflops\": \"$GFLOPS\" }" >> "$OUTPUT_FILE"
        done
        echo "        }" >> "$OUTPUT_FILE" # No comma after hw2_b_guided, as it's the last item.
        echo "      }" >> "$OUTPUT_FILE"
    done
    # CORRECTED: No comma after weak_scaling, as it's the last item in the profile.
    echo "    }" >> "$OUTPUT_FILE"
    echo "  }" >> "$OUTPUT_FILE"
done

echo "}" >> "$OUTPUT_FILE"

echo -e "\n--- Benchmark complete. Results saved to $OUTPUT_FILE ---"

echo "--- Generating PDF Report via report.py ---"
python3 report.py
if [ $? -ne 0 ]; then
    echo "Error: Report generation failed. Please check for errors from report.py."
    exit 1
fi
echo "--- Process Finished Successfully. Your report 'hw2.pdf' is ready. ---"

