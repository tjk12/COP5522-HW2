# Makefile for HW2 (Cross-Platform)

# --- Compiler and Flags ---
# Default to g++ for Linux. It can compile C++ code even in files named .c
CXX = g++
# Common flags: C++17 standard, all warnings, OpenMP support, link math library
COMMON_FLAGS = -std=c++17 -Wall -fopenmp -lm

# --- OS Detection for Compiler Selection ---
# On macOS, the default 'g++' is often an alias for 'clang', which may not have
# full OpenMP support. This logic finds a real GCC installed via Homebrew.
ifeq ($(shell uname -s), Darwin)
# Find all g++ versions in the common Homebrew path, sort them to get the latest
GCC_LIST := $(sort $(wildcard /usr/local/bin/g++-*))
# If the list is not empty, use the latest version
ifneq ($(GCC_LIST),)
CXX := $(lastword $(GCC_LIST))
else
# If no Homebrew GCC is found, stop and inform the user.
$(error No Homebrew GCC compiler (e.g., g++-13) found. Please install with 'brew install gcc'.)
endif
endif

# --- Optimization Profiles for Benchmarking ---
# These flag sets allow compiling with different optimization levels to compare
# their performance in the report.
# -O3: Aggressive speed optimization.
# -march=native: Allows the compiler to use all instructions supported by the local CPU.
# -mavx2 -mfma: Specifically enable AVX2 and Fused Multiply-Add instructions.
# -funroll-loops: An extra optimization to reduce loop overhead.
OPT_O3 = -O3 -march=native -mavx2 -mfma
OPT_O2 = -O2 -march=native -mavx2 -mfma
OPT_UNROLL = -O3 -march=native -mavx2 -mfma -funroll-loops

# --- File Definitions ---
TARGET_A = hw2-a
TARGET_B = hw2-b
# Source files are named .c as per the prompt, even though they contain C++ code.
SRC_A = hw2-a.cpp
SRC_B = hw2-b.cpp
TARGETS = $(TARGET_A) $(TARGET_B)

# --- Build Rules ---
# .PHONY declares targets that are not actual files.
.PHONY: all all-O3 all-O2 all-unroll clean test help

# Default target: `make` or `make all` will build with the -O3 profile.
all: all-O3

# Build with -O3 (default)
all-O3: CXXFLAGS = $(COMMON_FLAGS) $(OPT_O3)
all-O3: $(TARGETS)

# Build with -O2
all-O2: CXXFLAGS = $(COMMON_FLAGS) $(OPT_O2)
all-O2: $(TARGETS)

# Build with -O3 and unrolling
all-unroll: CXXFLAGS = $(COMMON_FLAGS) $(OPT_UNROLL)
all-unroll: $(TARGETS)

# Generic rule to build the executables.
# $< is the first prerequisite (the source file).
# $@ is the target name (the executable).
$(TARGET_A): $(SRC_A)
	@echo "Building '$@' with flags: $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_B): $(SRC_B)
	@echo "Building '$@' with flags: $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) -o $@ $<

# --- Testing ---
# A quick rule to compile and run a small test case.
test: all
	@echo "\n--- Running Quick Tests (N=1024, T=4) ---"
	OMP_NUM_THREADS=4 ./$(TARGET_A) 1024
	OMP_NUM_THREADS=4 ./$(TARGET_B) 1024 guided

# --- Cleanup ---
clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGETS) *.o

# --- Help ---
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Main Targets:"
	@echo "  all           Builds both executables with default -O3 optimizations (same as 'make all-O3')."
	@echo "  clean         Removes all compiled executables."
	@echo "  test          Runs a quick test with both executables."
	@echo "  help          Shows this help message."
	@echo ""
	@echo "Experimental Optimization Targets:"
	@echo "  all-O3        Builds using -O3, AVX2, and FMA flags."
	@echo "  all-O2        Builds using -O2, AVX2, and FMA flags."
	@echo "  all-unroll    Builds using -O3, AVX2, FMA, and -funroll-loops."
