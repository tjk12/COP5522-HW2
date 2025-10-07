# Makefile for OpenMP Matrix-Vector Multiplication Assignment (HW2) - C++ Version

# ---------------------------------
#  OS-AWARE COMPILER CONFIGURATION
# ---------------------------------
# Use 'uname' to detect the operating system for compiler selection.
UNAME_S := $(shell uname -s)

# If the OS is Darwin (macOS), automatically find the latest Homebrew GCC.
# Otherwise (e.g., on Linux), use the standard g++.
ifeq ($(UNAME_S),Darwin)
# --- macOS Compiler (Auto-detection) ---
# This command finds the latest version of g++ installed by Homebrew.
CXX := $(shell ls /usr/local/bin/g++-* 2>/dev/null | sort -V | tail -n 1)

# --- NEW: Check if the compiler was found and provide a helpful error ---
ifeq ($(strip $(CXX)),)
$(error No Homebrew GCC compiler (g++-*) found in /usr/local/bin/. Please run 'brew install gcc' to install it, then try again.)
endif
else
# --- Linux Compiler ---
CXX = g++
endif

# Common C++ flags: C++17 standard, high optimization, show all warnings, enable OpenMP.
# -march=native enables all instruction sets supported by the local machine.
# -mavx2 and -mfma are added for explicit compatibility and performance.
CXXFLAGS = -std=c++17 -O3 -Wall -fopenmp -march=native -mavx2 -mfma

# Linker flags (e.g., for math library)
LDFLAGS = -lm

# ---------------------------------
#         FILE DEFINITIONS
# ---------------------------------
# Define executables
TARGETS = hw2-a hw2-b

# ---------------------------------
#           BUILD RULES
# ---------------------------------
# Phony targets are not files.
.PHONY: all clean test run

# Default target: 'make' or 'make all' will build both executables.
all: $(TARGETS)

# Rule to build the dense matrix executable from the C++ source
hw2-a: hw2-a.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built dense matrix executable using '$(CXX)': ./hw2-a <size>"

# Rule to build the triangular matrix executable from C++ source
hw2-b: hw2-b.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built triangular matrix executable using '$(CXX)': ./hw2-b <size> <schedule>"


# ---------------------------------
#        TESTING & RUNNING
# ---------------------------------
# 'make test' runs a quick check with small inputs
test: all
	@echo "\n--- Running Quick Test (N=1024, 4 Threads) ---"
	@export OMP_NUM_THREADS=4 && echo "Running hw2-a (dense, C++)..." && ./hw2-a 1024
	@export OMP_NUM_THREADS=4 && echo "\nRunning hw2-b (triangular, C++)..." && ./hw2-b 1024 guided
	@echo "\n--- Quick Test Complete ---"

# 'make run' executes the full benchmark script
run: all
	@echo "Running full benchmark script..."
	@chmod +x run_benchmarks.sh
	@./run_benchmarks.sh

# ---------------------------------
#            CLEANUP
# ---------------------------------
# 'make clean' removes all generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGETS) *.o results.json hw2.pdf *.png

# ---------------------------------
#              HELP
# ---------------------------------
help:
	@echo "Available targets:"
	@echo "  all    - Build both hw2-a and hw2-b executables from C++ sources"
	@echo "  test   - Run a small test case for both programs"
	@echo "  run    - Run the full benchmark script to generate results.json"
	@echo "  clean  - Remove all compiled files and generated reports"

