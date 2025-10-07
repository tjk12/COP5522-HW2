import json
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from collections import defaultdict
import tarfile

# --- Configuration ---
RESULTS_FILE = "results.json"
PDF_FILE = "hw2.pdf"
LOG_FILE = "ai-usage.txt"
CHART_STRONG_SCALING = "strong_scaling.png"
CHART_WEAK_SCALING = "weak_scaling.png"

def generate_report(data):
    """Generates the full PDF report from the structured benchmark data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # --- Title ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "HW2 Performance Analysis: OpenMP Matrix-Vector Multiplication", ln=True, align="C")
    pdf.ln(5)

    # --- Introduction ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Introduction", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, 
        "This report analyzes the performance of two parallel matrix-vector multiplication algorithms implemented using OpenMP: "
        "a standard dense matrix multiplication (hw2-a) and a specialized version for lower-triangular matrices (hw2-b). "
        "The analysis focuses on comparing different compiler optimization strategies and evaluating the strong and weak scaling characteristics of the parallel implementations.")
    pdf.ln(5)

    # --- Compiler Optimization Strategies ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Compiler Optimization Strategies", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "To evaluate the impact of compiler settings as requested, several optimization profiles were tested. All profiles use '-march=native -mavx2 -mfma' to enable modern CPU vector instructions.")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "O2 Optimized:", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, "Uses the '-O2' flag, a standard and stable level of optimization that balances code size and performance.")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "O3 Unrolled:", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, "Adds the '-funroll-loops' flag to the '-O3' profile, which can improve performance by reducing loop overhead at the cost of a larger binary size.")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "O3 Default:", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, "Uses the '-O3' flag, which enables a high level of aggressive optimizations focused on execution speed.")    
    pdf.ln(5)

    # --- Analysis of hw2-b Optimizations ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Comparison of Compiler Optimizations for hw2-b", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "The following table compares the performance of the triangular matrix-vector multiplication (hw2-b) "
        "across the different compiler profiles. For each matrix size, the table shows the peak performance in Gflop/s and, in parentheses, the optimal thread count. "
        "The best compiler optimization for each matrix size is highlighted.")
    pdf.ln(5)
    table_data = create_optimization_table(pdf, data)
    pdf.ln(10)

    # --- Scaling Analysis ---
    best_profile_key = find_best_profile_by_wins(data, table_data)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"4. Scaling Analysis (using '{best_profile_key}' profile)", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, f"The following graphs illustrate strong and weak scaling performance for both the dense (hw2-a) and triangular (hw2-b) implementations. These results were generated using the '{best_profile_key}' compilation profile, which demonstrated the most consistent high performance in the tests above.")
    pdf.ln(5)

    if best_profile_key and best_profile_key in data:
        profile_data = data[best_profile_key]
        generate_scaling_chart(profile_data.get('strong_scaling', {}), "Strong Scaling Comparison", "Threads", "Performance (Gflop/s)", CHART_STRONG_SCALING)
        generate_scaling_chart(profile_data.get('weak_scaling', {}), "Weak Scaling Comparison", "Threads", "Performance (Gflop/s)", CHART_WEAK_SCALING, is_weak=True)
        if os.path.exists(CHART_STRONG_SCALING): pdf.image(CHART_STRONG_SCALING, x=10, y=None, w=180)
        if os.path.exists(CHART_WEAK_SCALING): pdf.image(CHART_WEAK_SCALING, x=10, y=None, w=180)
    else:
        pdf.cell(0, 10, "Could not generate scaling charts: Best profile data not found.", ln=True)

    pdf.add_page() 
    
    # --- Analysis of Scaling Performance ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "5. Analysis of Scaling Performance", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "Strong Scaling Insights", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, 
        "Strong scaling measures how the execution time varies for a fixed total problem size as the number of threads increases. "
        "Ideally, the performance (Gflop/s) should increase linearly with the number of threads, as shown by the dashed line in the plot. However, the observed results typically show a curve that flattens out at higher thread counts. "
        "This is explained by Amdahl's Law, which states that the maximum speedup is limited by the sequential portion of the code (e.g., memory allocation, initialization, final reduction steps) and the overhead of managing the parallel threads. "
        "For hw2-b, the irregular workload (each row has a different number of non-zero elements) can further impact scaling, as some threads may finish before others, especially with static scheduling.")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "Weak Scaling Insights", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, 
        "Weak scaling measures how the execution time varies as both the problem size and the number of threads increase proportionally (i.e., the work per thread remains constant). "
        "In an ideal scenario, the execution time would remain constant, and therefore the performance in Gflop/s should increase linearly with the number of threads. "
        "In practice, performance often falls short of this ideal linear scaling. This is typically due to system-level bottlenecks that become more pronounced as the total problem size grows, such as increased contention for shared memory bandwidth or limitations in cache capacity. "
        "The dense matrix (hw2-a) generally scales better in weak scaling tests due to its highly regular memory access patterns, whereas hw2-b's irregular access might lead to less predictable cache performance.")
    pdf.ln(10)

    # --- Reflection on AI Tool Usage ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "6. Reflection on AI Tool Usage", ln=True)
    pdf.set_font("Helvetica", size=10)
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
            if "### Reflection on AI Tool Usage" in log_content:
                reflection_text = log_content.split("### Reflection on AI Tool Usage")[1]
                pdf.multi_cell(0, 5, reflection_text.strip())
            else: pdf.multi_cell(0, 5, log_content)
    except FileNotFoundError:
        pdf.multi_cell(0, 5, f"Error: {LOG_FILE} not found. Could not include AI reflection.")
    except IndexError:
        pdf.multi_cell(0, 5, f"Error: Could not find '### Reflection on AI Tool Usage' section in {LOG_FILE}.")

    pdf.output(PDF_FILE)
    print(f"Report successfully generated: {PDF_FILE}")
    if os.path.exists(CHART_STRONG_SCALING): os.remove(CHART_STRONG_SCALING)
    if os.path.exists(CHART_WEAK_SCALING): os.remove(CHART_WEAK_SCALING)

def get_peak_performance_for_size(profile_data, size):
    best_gflops, best_threads = 0.0, 0
    try:
        schedules_data = profile_data['general_perf']['hw2_b'][size]
        for schedule in schedules_data:
            for thread, gflops in schedules_data[schedule].items():
                try: gflops_val = float(gflops)
                except (ValueError, TypeError): gflops_val = 0.0
                if gflops_val > best_gflops:
                    best_gflops, best_threads = gflops_val, int(thread[1:])
    except (KeyError, TypeError): pass
    return {'gflops': best_gflops, 'threads': best_threads}

def create_optimization_table(pdf, data):
    if not data:
        pdf.cell(0, 10, "No data available to generate table.", ln=True)
        return {}
    profile_keys = list(data.keys())
    table_data = defaultdict(dict)
    pdf.set_font("Helvetica", "B", 10)
    col_width = 180 / (len(profile_keys) + 1)
    header = ["Matrix Size"] + [k.replace("_", " ").title() for k in profile_keys]
    for item in header:
        pdf.cell(col_width, 10, item, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", size=9)
    try:
        matrix_sizes = sorted(data[profile_keys[0]]['general_perf']['hw2_b'].keys(), key=lambda x: int(x[1:]))
    except (KeyError, IndexError, TypeError):
        pdf.cell(0, 10, "No valid hw2-b performance data found to build table.", ln=True)
        return {}
    for size in matrix_sizes:
        row_data = [get_peak_performance_for_size(data.get(p, {}), size) for p in profile_keys]
        for i, p in enumerate(profile_keys):
            table_data[size][p] = row_data[i]
        max_gflops_in_row = max(d['gflops'] for d in row_data) if row_data else 0.0
        pdf.cell(col_width, 10, f"{size[1:]}x{size[1:]}", border=1, align="C")
        for cell_data in row_data:
            is_best = cell_data['gflops'] == max_gflops_in_row and max_gflops_in_row > 0.0
            if is_best: pdf.set_fill_color(200, 220, 255)
            cell_text = f"{cell_data['gflops']:.2f} ({cell_data['threads']}T)" if cell_data['gflops'] > 0 else "N/A"
            pdf.cell(col_width, 10, cell_text, border=1, align="C", fill=is_best)
        pdf.ln()
    return table_data

def find_best_profile_by_wins(data, table_data):
    if not table_data: return list(data.keys())[0] if data else "O3_default"
    win_counts, peak_gflops = defaultdict(int), defaultdict(float)
    for size, profiles in table_data.items():
        max_gflops_in_row = max(p['gflops'] for p in profiles.values()) if profiles else 0.0
        if max_gflops_in_row > 0:
            for profile, perf in profiles.items():
                if perf['gflops'] == max_gflops_in_row: win_counts[profile] += 1
                if perf['gflops'] > peak_gflops[profile]: peak_gflops[profile] = perf['gflops']
    if not win_counts: return list(data.keys())[0] if data else "O3_default"
    return sorted(win_counts.keys(), key=lambda p: (win_counts[p], peak_gflops[p]), reverse=True)[0]

def generate_scaling_chart(data, title, xlabel, ylabel, filename, is_weak=False):
    if not data: return
    plt.figure(figsize=(10, 5.5))
    plot_keys = {'hw2_a': 'Dense (hw2-a)', 'hw2_b_guided': 'Triangular (hw2-b)'}
    
    # Plot real data
    for key, label in plot_keys.items():
        if key not in data: continue
        results = data[key]
        try:
            threads_int = sorted([int(t[1:]) for t in results.keys()])
            threads_str = [f"T{t}" for t in threads_int]
            if is_weak:
                gflops = [float(results[t]['gflops']) for t in threads_str if results.get(t) and results[t].get('gflops')]
                threads_plot = [int(t[1:]) for t in threads_str if results.get(t) and results[t].get('gflops')]
            else:
                gflops = [float(results[t]) for t in threads_str if results.get(t)]
                threads_plot = [int(t[1:]) for t in threads_str if results.get(t)]
            if threads_plot: plt.plot(threads_plot, gflops, marker='o', linestyle='-', label=label)
        except (ValueError, TypeError, KeyError) as e:
            print(f"Warning: Could not plot data for '{label}' in '{title}'. Error: {e}")
            continue

    # --- Add Ideal Scaling Line ---
    ideal_base_key = 'hw2_a'
    if ideal_base_key in data and 'T1' in data[ideal_base_key]:
        try:
            if is_weak:
                t1_gflops = float(data[ideal_base_key]['T1']['gflops'])
                threads_int_ideal = sorted([int(t[1:]) for t in data[ideal_base_key].keys()])
                ideal_gflops = [t1_gflops * t for t in threads_int_ideal]
                if threads_int_ideal:
                    plt.plot(threads_int_ideal, ideal_gflops, linestyle='--', color='k', label='Ideal Linear Scaling')
            else:
                t1_gflops = float(data[ideal_base_key]['T1'])
                threads_int_ideal = sorted([int(t[1:]) for t in data[ideal_base_key].keys()])
                ideal_gflops = [t1_gflops * t for t in threads_int_ideal]
                if threads_int_ideal:
                    plt.plot(threads_int_ideal, ideal_gflops, linestyle='--', color='k', label='Ideal Linear Scaling')
        except (ValueError, TypeError, KeyError) as e:
            print(f"Warning: Could not plot ideal scaling line. Base 'T1' data may be missing or invalid. Error: {e}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plt.gca().has_data(): plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_submission_archive():
    """Creates a tar archive with all necessary submission files."""
    archive_name = "hw2.tar"
    files_to_archive = [
        "hw2-a.cpp",
        "hw2-b.cpp",
        "hw2.pdf",
        "LOG.txt",
        "makefile",
    ]
    
    print("\n--- Creating Submission Archive ---")
    
    try:
        with tarfile.open(archive_name, "w") as tar:
            for filename in files_to_archive:
                if os.path.exists(filename):
                    tar.add(filename)
                    print(f"  Adding {filename} to archive.")
                else:
                    print(f"  Warning: {filename} not found, skipping.")
        
        print(f"Successfully created submission archive: {archive_name}")
    except Exception as e:
        print(f"Error: Failed to create tar archive. Reason: {e}")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run './run_benchmarks.sh' first.")
    else:
        with open(RESULTS_FILE, "r") as f:
            try:
                benchmark_results = json.load(f)
                generate_report(benchmark_results)
                create_submission_archive()
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {RESULTS_FILE}. It may be corrupted.")

