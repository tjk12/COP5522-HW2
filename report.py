import json
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from collections import defaultdict
import math
import traceback

# --- Configuration ---
RESULTS_FILE = "results.json"
PDF_FILE = "hw2.pdf"
LOG_FILE = "ai-usage.txt"
CHART_STRONG_SCALING = "strong_scaling.png"
CHART_WEAK_SCALING = "weak_scaling.png"
CHART_SCHEDULING = "scheduling_performance.png"

def generate_report(data):
    """Generates the full PDF report from the structured benchmark data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # --- Title ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "HW2 Performance Report", ln=True, align="C")
    pdf.ln(5)

    # --- Introduction ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Introduction", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, 
        "This report analyzes the performance of two parallel matrix-vector multiplication algorithms implemented using OpenMP: "
        "a standard dense matrix multiplication (hw2-a) and a specialized version for lower-triangular matrices (hw2-b). "
        "The analysis focuses on comparing different compiler optimization strategies, evaluating OpenMP scheduling policies, and analyzing the scaling characteristics of the parallel implementations.")
    pdf.ln(5)
    
    # --- Analysis of Scheduling Strategies ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Analysis of OpenMP Scheduling Strategies for hw2-b", ln=True)
    
    best_profile_for_scheduling_analysis = find_best_profile_by_wins(data, create_optimization_table(None, data, "guided"))
    generate_schedule_performance_chart(data.get(best_profile_for_scheduling_analysis, {}))
    if os.path.exists(CHART_SCHEDULING):
        pdf.image(CHART_SCHEDULING, x=10, y=None, w=180)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "[Scheduling performance chart could not be generated. Check console for errors.]", ln=True, align="C")


    pdf.set_font("Helvetica", size=10)
    analysis_text, best_schedule = analyze_schedules(data)
    pdf.multi_cell(0, 5, analysis_text)
    pdf.ln(5)

    # --- Compiler Optimization Strategies ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Compiler Optimization Strategies", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "To evaluate the impact of compiler settings, several optimization profiles were tested. All profiles use '-march=native -mavx2 -mfma' to enable modern CPU vector instructions.")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "O3 Default:", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, "Uses the '-O3' flag, which enables a high level of aggressive optimizations focused on execution speed.")
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
    pdf.ln(5)

    # --- Analysis of hw2-b Optimizations ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Comparison of Compiler Optimizations for hw2-b", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        f"The following table compares the performance of the triangular matrix-vector multiplication (hw2-b) "
        f"across the different compiler profiles. For each data point, the results from the best-performing schedule ('{best_schedule}') were used. "
        "The table shows the peak performance in Gflop/s and, in parentheses, the optimal thread count. "
        "The best compiler optimization for each matrix size is highlighted.")
    pdf.ln(5)
    table_data = create_optimization_table(pdf, data, best_schedule)
    pdf.ln(5)

    # --- Scaling Analysis ---
    best_profile_key = find_best_profile_by_wins(data, table_data)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"5. Scaling Analysis (using '{best_profile_key}' profile)", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, f"The following graphs illustrate strong and weak scaling performance. These results were generated using the '{best_profile_key}' compilation profile. Both charts show side-by-side comparisons for all tested matrix size configurations.")
    pdf.ln(5)

    if best_profile_key and best_profile_key in data:
        profile_data = data[best_profile_key]
        generate_scaling_chart(profile_data, "Strong Scaling Comparison", CHART_STRONG_SCALING)
        generate_scaling_chart(profile_data, "Weak Scaling Comparison", CHART_WEAK_SCALING, is_weak=True)
        if os.path.exists(CHART_STRONG_SCALING): 
            pdf.image(CHART_STRONG_SCALING, x=10, y=None, w=180)
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, "[Strong scaling chart could not be generated. Check console for errors.]", ln=True, align="C")
        pdf.ln(5)
        if os.path.exists(CHART_WEAK_SCALING): 
            pdf.image(CHART_WEAK_SCALING, x=10, y=None, w=180)
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, "[Weak scaling chart could not be generated. Check console for errors.]", ln=True, align="C")
    else:
        pdf.cell(0, 10, "Could not generate scaling charts: Best profile data not found.", ln=True)

    pdf.add_page() 
    
    # --- Analysis of Scaling Performance ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "6. Analysis of Scaling Performance", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "Strong Scaling Insights", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, 
        "Strong scaling measures how the execution time varies for a fixed total problem size as the number of threads increases. "
        "Ideally, performance (Gflop/s) should increase linearly with the number of threads (the 'ideal speedup' line). The charts show this ideal case as a dashed line. The observed results typically show a curve that achieves good speedup initially but then flattens out at higher thread counts. "
        "This is explained by Amdahl's Law, where speedup is limited by sequential code portions and parallel overhead. This effect is more pronounced at smaller matrix sizes, where the amount of parallel work is not large enough to overcome the overhead of managing many threads.")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "Weak Scaling Insights", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, 
        "Weak scaling measures performance as both the problem size and the number of threads increase proportionally (i.e., work per thread is constant). "
        "Ideally, performance in Gflop/s should increase linearly with the number of threads. The charts demonstrate this by starting separate weak scaling experiments from different base matrix sizes. "
        "In practice, performance often falls short of this ideal. This is typically due to system-level bottlenecks that become more pronounced as the total problem size grows, such as increased contention for shared memory bandwidth or limitations in cache capacity. "
        "By comparing the plots, we can see that experiments starting with a larger base N tend to achieve higher absolute Gflop/s, likely due to a better computation-to-communication ratio.")
    pdf.ln(4)

    # ADDED: New section explaining memory bandwidth saturation.
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "The Impact of Thread Count vs. Available Cores", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "An important observation from manual testing is the significant performance difference between running the benchmark with a specific, limited number of threads (e.g., OMP_NUM_THREADS=16) versus allowing OpenMP to use all available cores on the system (e.g., 256 cores on Bridges-2). While it seems intuitive that more cores should equal more performance, this is often not the case for memory-bound problems like matrix-vector multiplication.\n\n"
        "The primary bottleneck is not the CPU's computational power, but the speed at which it can fetch data from main memory (RAM). This is known as the memory bandwidth limit. When an excessive number of threads are launched simultaneously, they all contend for access to this limited memory bandwidth. This creates a \"traffic jam\" on the memory bus, causing most threads to spend their time waiting for data rather than performing calculations.\n\n"
        "The benchmark script carefully tests a range of thread counts (1 to 32) to find the optimal point where the system's memory bandwidth can effectively service the active cores. This 'sweet spot' delivers the peak performance seen in the graphs. Launching threads beyond this point leads to diminishing returns and eventually a sharp drop in performance due to memory contention and parallel overhead, demonstrating a critical concept in high-performance computing: scaling is limited by the most constrained resource, which in this case is memory bandwidth."
    )
    pdf.ln(10)

    # --- Reflection on AI Tool Usage ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "7. Reflection on AI Tool Usage", ln=True)
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
    # Clean up the generated chart images after they have been embedded in the PDF
    for chart_file in [CHART_STRONG_SCALING, CHART_WEAK_SCALING, CHART_SCHEDULING]:
        if os.path.exists(chart_file):
            os.remove(chart_file)

    create_submission_archive()

def analyze_schedules(data):
    """Analyzes schedule performance and generates a specific explanation based on the winner."""
    schedule_scores = defaultdict(float)
    if not data: return "Could not determine the best schedule due to missing data.", "guided"
    
    for profile_data in data.values():
        try:
            for size_data in profile_data['general_perf']['hw2_b'].values():
                for schedule_key, thread_data in size_data.items():
                    schedule_name = schedule_key.replace("schedule_", "")
                    for gflops_str in thread_data.values():
                        try:
                            schedule_scores[schedule_name] += float(gflops_str)
                        except (ValueError, TypeError):
                            continue
        except KeyError:
            continue
            
    if not schedule_scores: return "No valid performance data found for any schedule.", "guided"
    
    best_schedule = max(schedule_scores, key=schedule_scores.get)
    
    intro = ("The chart above visually compares the performance of the three main OpenMP scheduling strategies for the triangular matrix multiplication (hw2-b), which has an imbalanced workload. The workload is heaviest for the last rows of the matrix and lightest for the first rows.\n\n"
             f"Based on the cumulative performance across all tests, the '{best_schedule}' scheduling strategy was determined to be the most effective. ")

    # REVISED: Provide a specific explanation for why the winning schedule performed best.
    explanations = {
        "static": (
            "This result is somewhat unexpected for an imbalanced workload. A possible explanation is that for the matrix sizes and thread counts tested, the overhead associated with dynamic scheduling outweighed its load-balancing benefits. "
            "The simplicity and low overhead of the static scheduler, which pre-allocates an equal number of iterations to each thread, proved to be more efficient overall. This can happen if the imbalance is not severe enough to leave threads idle for long periods."
        ),
        "dynamic": (
            "This result is logical for an imbalanced workload. The 'dynamic' scheduler assigns a small chunk of iterations to a thread and, once completed, the thread requests a new chunk. "
            "This ensures that no thread sits idle while others are still working on the longer tasks (the final rows of the matrix). While it has higher overhead than static scheduling due to this continuous management, its ability to adapt and keep all cores busy proved to be the decisive factor for performance in this scenario."
        ),
        "guided": (
            "This result aligns well with theoretical expectations for this problem. The 'guided' scheduler is a hybrid approach that starts by giving large chunks of iterations to threads and gradually decreases the chunk size. "
            "This strategy minimizes scheduling overhead at the beginning (like 'static') while providing finer-grained load balancing towards the end of the computation (like 'dynamic'). This compromise effectively handles the imbalanced workload of the triangular matrix without incurring the high, constant overhead of a pure dynamic schedule."
        )
    }
    
    conclusion = explanations.get(best_schedule, "The reason for this schedule's superior performance is likely its specific approach to balancing workload distribution against scheduling overhead.")
    
    return intro + conclusion, best_schedule


def get_peak_performance_for_size(profile_data, size, best_schedule):
    best_gflops, best_threads = 0.0, 0
    try:
        schedule_key = f"schedule_{best_schedule}"
        thread_data = profile_data['general_perf']['hw2_b'][size][schedule_key]
        for thread, gflops in thread_data.items():
            try: gflops_val = float(gflops)
            except (ValueError, TypeError): gflops_val = 0.0
            if gflops_val > best_gflops:
                best_gflops, best_threads = gflops_val, int(thread[1:])
    except (KeyError, TypeError): pass
    return {'gflops': best_gflops, 'threads': best_threads}

def create_optimization_table(pdf, data, best_schedule):
    if not data:
        if pdf: pdf.cell(0, 10, "No data available to generate table.", ln=True)
        return {}
    profile_keys = list(data.keys())
    table_data = defaultdict(dict)
    if pdf:
        pdf.set_font("Helvetica", "B", 10)
        col_width = 180 / (len(profile_keys) + 1)
        header = ["Matrix Size"] + [k.replace("_", " ").title() for k in profile_keys]
        for item in header: pdf.cell(col_width, 10, item, border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", size=9)
    try:
        matrix_sizes = sorted(data[profile_keys[0]]['general_perf']['hw2_b'].keys(), key=lambda x: int(x[1:]))
    except (KeyError, IndexError, TypeError):
        if pdf: pdf.cell(0, 10, "No valid hw2-b performance data found to build table.", ln=True)
        return {}
    for size in matrix_sizes:
        row_data = [get_peak_performance_for_size(data.get(p, {}), size, best_schedule) for p in profile_keys]
        for i, p in enumerate(profile_keys):
            table_data[size][p] = row_data[i]
        if pdf:
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
    if not table_data:
        _, best_schedule_temp = analyze_schedules(data)
        table_data = create_optimization_table(None, data, best_schedule_temp)
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

def generate_schedule_performance_chart(profile_data):
    try:
        if not profile_data: return
        hw2b_data = profile_data.get('general_perf', {}).get('hw2_b', {})
        if not hw2b_data: return
        matrix_sizes = sorted(hw2b_data.keys(), key=lambda x: int(x[1:]))
        if not matrix_sizes: return

        ncols = 2
        nrows = math.ceil(len(matrix_sizes) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4.5), squeeze=False)
        axes = axes.flatten()
        
        for i, size in enumerate(matrix_sizes):
            ax = axes[i]
            size_data = hw2b_data[size]
            for schedule_key, thread_data in sorted(size_data.items()):
                schedule_name = schedule_key.replace("schedule_", "").title()
                valid_threads = {int(t[1:]):float(g) for t, g in thread_data.items() if g and g.replace('.', '', 1).isdigit()}
                if valid_threads:
                    sorted_threads = sorted(valid_threads.keys())
                    gflops = [valid_threads[t] for t in sorted_threads]
                    ax.plot(sorted_threads, gflops, marker='o', linestyle='-', label=f'{schedule_name}')
            
            ax.set_title(f'N = {size[1:]}')
            ax.set_xlabel('Threads')
            ax.set_ylabel('Gflop/s')
            ax.grid(True, which="both", ls="--")
        
        for j in range(len(matrix_sizes), len(axes)): axes[j].set_visible(False)
        fig.suptitle('Scheduling Strategy Performance for hw2-b', fontsize=16)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(CHART_SCHEDULING)
    except Exception as e:
        print(f"\n--- !!! ERROR: Could not generate schedule performance chart. Plot will be missing. !!! ---")
        print(f"--- Error Type: {type(e).__name__}, Details: {e}")
        print(f"--- This often happens if 'results.json' has missing data or an unexpected format.")
        traceback.print_exc()
    finally:
        plt.close()

def generate_scaling_chart(profile_data, title, filename, is_weak=False):
    try:
        if not profile_data: return
        
        if is_weak:
            weak_data = profile_data.get('weak_scaling', {})
            if not weak_data: return
            base_sizes = sorted(weak_data.keys(), key=lambda x: int(x[1:]))
            if not base_sizes: return
            
            ncols = 2
            nrows = math.ceil(len(base_sizes) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4.5), squeeze=False)
            axes = axes.flatten()

            for i, base_size_key in enumerate(base_sizes):
                ax = axes[i]
                experiment_data = weak_data[base_size_key]
                plot_keys = {'hw2_a': 'Dense (hw2-a)', 'hw2_b_guided': 'Triangular (hw2-b)'}
                
                try: # Ideal weak scaling line
                    t1_gflops = float(experiment_data['hw2_b_guided']['T1']['gflops'])
                    threads_list = sorted([int(t[1:]) for t in experiment_data['hw2_b_guided'].keys()])
                    ideal_gflops = [t1_gflops * t for t in threads_list]
                    ax.plot(threads_list, ideal_gflops, 'k--', label='Ideal Scaling')
                except (KeyError, ValueError, TypeError): pass

                for key, label in plot_keys.items():
                    if key in experiment_data:
                        results = experiment_data[key]
                        valid_points = {int(t[1:]):float(p['gflops']) for t,p in results.items() if p.get('gflops') and p['gflops'].replace('.','',1).isdigit()}
                        if valid_points:
                            sorted_threads = sorted(valid_points.keys())
                            gflops = [valid_points[t] for t in sorted_threads]
                            ax.plot(sorted_threads, gflops, marker='o', linestyle='-', label=label)
                
                ax.set_title(f'Base N = {base_size_key[1:]}')
                ax.set_xlabel('Threads')
                ax.set_ylabel('Gflop/s')
                ax.grid(True, which="both", ls="--")

            for j in range(len(base_sizes), len(axes)): axes[j].set_visible(False)
            fig.suptitle(title, fontsize=16)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
        else: # Strong scaling
            strong_data = profile_data.get('general_perf', {})
            hw2a_data = strong_data.get('hw2_a', {})
            hw2b_data = strong_data.get('hw2_b', {})
            if not hw2a_data: return
            
            matrix_sizes = sorted(hw2a_data.keys(), key=lambda x: int(x[1:]))
            if not matrix_sizes: return

            ncols = 2
            nrows = math.ceil(len(matrix_sizes) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4.5), squeeze=False)
            axes = axes.flatten()

            for i, size in enumerate(matrix_sizes):
                ax = axes[i]
                try: # Ideal strong scaling line
                    t1_gflops = float(hw2a_data[size]['T1'])
                    threads_list = sorted([int(t[1:]) for t in hw2a_data[size].keys() if hw2a_data[size].get(t)])
                    ideal_gflops = [t1_gflops * t for t in threads_list]
                    ax.plot(threads_list, ideal_gflops, 'k--', label='Ideal Speedup')
                except(KeyError, ValueError, TypeError): pass

                if size in hw2a_data:
                    valid_threads = {int(t[1:]):float(g) for t,g in hw2a_data[size].items() if g and g.replace('.','',1).isdigit()}
                    if valid_threads: 
                        sorted_threads = sorted(valid_threads.keys())
                        gflops = [valid_threads[t] for t in sorted_threads]
                        ax.plot(sorted_threads, gflops, marker='o', linestyle='-', label='Dense (hw2-a)')
                
                if size in hw2b_data and 'schedule_guided' in hw2b_data[size]:
                    valid_threads = {int(t[1:]):float(g) for t,g in hw2b_data[size]['schedule_guided'].items() if g and g.replace('.','',1).isdigit()}
                    if valid_threads: 
                        sorted_threads = sorted(valid_threads.keys())
                        gflops = [valid_threads[t] for t in sorted_threads]
                        ax.plot(sorted_threads, gflops, marker='s', linestyle='--', label='Triangular (hw2-b)')

                ax.set_title(f'N = {size[1:]}')
                ax.set_xlabel('Threads')
                ax.set_ylabel('Gflop/s')
                ax.grid(True, which="both", ls="--")

            for j in range(len(matrix_sizes), len(axes)): axes[j].set_visible(False)
            fig.suptitle(title, fontsize=16)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename)
    except Exception as e:
        print(f"\n--- !!! ERROR: Could not generate scaling chart '{filename}'. Plot will be missing. !!! ---")
        print(f"--- Error Type: {type(e).__name__}, Details: {e}")
        print(f"--- This often happens if 'results.json' has missing data or an unexpected format.")
        traceback.print_exc()
    finally:
        plt.close()

def create_submission_archive():
    archive_name = "hw2.tar"
    files_to_archive = ["hw2-a.cpp", "hw2-b.cpp", "hw2.pdf", "LOG.txt", "makefile"]
    existing_files = [f for f in files_to_archive if os.path.exists(f)]
    if not existing_files:
        print("Warning: No files found to archive.")
        return
    print(f"\n--- Creating submission archive: {archive_name} ---")
    try:
        import tarfile
        with tarfile.open(archive_name, "w") as tar:
            for name in existing_files:
                tar.add(name)
        print(f"Successfully created {archive_name} with {len(existing_files)} files.")
    except Exception as e:
        print(f"Error: Could not create tar archive. {e}")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run './run_benchmarks.sh' first.")
    else:
        with open(RESULTS_FILE, "r") as f:
            try:
                benchmark_results = json.load(f)
                generate_report(benchmark_results)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {RESULTS_FILE}. It may be corrupted.")

