# Makefile for HW2 (Cross-Platform)

# --- Compiler and Flags ---
# Default to g++ for Linux
CXX = g++
# Common flags: C++17, Wall for warnings, OpenMP support, lm for the math library
COMMON_FLAGS = -std=c++17 -Wall -fopenmp -lm

# --- OS Detection for Compiler Selection ---
# Use 'uname -s' to check the operating system kernel name
ifeq ($(shell uname -s), Darwin)
# macOS: Find the latest Homebrew GCC compiler
# The 'wildcard' function finds matching files, 'sort' gets the latest version.
GCC_LIST := $(sort $(wildcard /usr/local/bin/g++-*))
# Check if any GCC versions were found
ifneq ($(GCC_LIST),)
# Select the last one in the sorted list (latest version)
CXX := $(lastword $(GCC_LIST))
else
# If no GCC found, stop with a helpful error message
$(error No Homebrew GCC compiler (e.g., g++-13) found in /usr/local/bin/. Please install it with 'brew install gcc'.)
endif
endif

# --- Experimental Optimization Profiles ---
# These flags are appended to COMMON_FLAGS based on the make target
OPT_O3 = -O3 -march=native -mavx2 -mfma
OPT_O2 = -O2 -march=native -mavx2 -mfma
OPT_UNROLL = -O3 -march=native -mavx2 -mfma -funroll-loops

# --- File Definitions ---
TARGET_A = hw2-a
TARGET_B = hw2-b
SRC_A = hw2-a.cpp
SRC_B = hw2-b.cpp
TARGETS = $(TARGET_A) $(TARGET_B)

# --- Build Rules ---
.PHONY: all all-O3 all-O2 all-unroll clean test help

# Default target: build with -O3 optimizations
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

# Generic rule to build the executables
# Use a TAB character for indentation on the command lines below
$(TARGET_A): $(SRC_A)
	@echo "Building '$@' with flags: $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_B): $(SRC_B)
	@echo "Building '$@' with flags: $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) -o $@ $<

# --- Testing ---
test: all
	@echo "\n--- Running Quick Tests (N=1024, T=4) ---"
	OMP_NUM_THREADS=4 ./$(TARGET_A) 1024
	OMP_NUM_THREADS=4 ./$(TARGET_B) 1024 static

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

