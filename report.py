import json
import os
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- Configuration ---
RESULTS_FILE = "results.json"
PDF_FILE = "hw2.pdf"
LOG_FILE = "LOG.txt"
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
    pdf.ln(10)

    # --- Analysis of hw2-b Optimizations ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Comparison of Compiler Optimizations for hw2-b", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "The following table compares the performance of the triangular matrix-vector multiplication (hw2-b) "
        "across different compiler optimization profiles. For each matrix size and optimization strategy, "
        "the table shows the peak performance achieved in Gflop/s and, in parentheses, the number of threads "
        "that yielded this result. The best-performing scheduling strategy (static, dynamic, or guided) was "
        "automatically selected for each data point. The best result for each matrix size is highlighted.")
    pdf.ln(5)

    # --- Generate the new optimization comparison table ---
    create_optimization_table(pdf, data)
    pdf.ln(10)

    # --- Scaling Analysis ---
    # Find the best overall optimization profile to use for the scaling graphs
    best_profile_key = find_best_profile(data)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"2. Scaling Analysis (using '{best_profile_key}' profile)", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, f"The following graphs illustrate strong and weak scaling performance. These results were generated using the '{best_profile_key}' compilation profile, which demonstrated the best overall performance in the tests.")
    pdf.ln(5)

    # Generate and embed scaling charts
    if best_profile_key and best_profile_key in data:
        strong_scaling_data = data[best_profile_key].get('strong_scaling', {})
        weak_scaling_data = data[best_profile_key].get('weak_scaling', {})
        generate_scaling_chart(strong_scaling_data, "Strong Scaling", "Threads", "Performance (Gflop/s)", CHART_STRONG_SCALING)
        generate_scaling_chart(weak_scaling_data, "Weak Scaling", "Threads", "Performance (Gflop/s)", CHART_WEAK_SCALING, is_weak=True)

        if os.path.exists(CHART_STRONG_SCALING):
            pdf.image(CHART_STRONG_SCALING, x=10, y=None, w=180)
        if os.path.exists(CHART_WEAK_SCALING):
            pdf.image(CHART_WEAK_SCALING, x=10, y=None, w=180)
    else:
        pdf.cell(0, 10, "Could not generate scaling charts: Best profile data not found.", ln=True)

    pdf.add_page() # New page for AI reflection

    # --- AI Reflection ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Reflection on AI Tool Usage", ln=True)
    pdf.set_font("Helvetica", size=10)
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
            # Extract only the reflection part if the keyword exists
            if "### Reflection on AI Tool Usage" in log_content:
                reflection_text = log_content.split("### Reflection on AI Tool Usage")[1]
                pdf.multi_cell(0, 5, reflection_text.strip())
            else:
                pdf.multi_cell(0, 5, log_content) # Use full content if keyword is missing
    except FileNotFoundError:
        pdf.multi_cell(0, 5, f"Error: {LOG_FILE} not found. Could not include AI reflection.")
    except IndexError:
        pdf.multi_cell(0, 5, f"Error: Could not find '### Reflection on AI Tool Usage' section in {LOG_FILE}.")

    # --- Save the PDF ---
    pdf.output(PDF_FILE)
    print(f"Report successfully generated: {PDF_FILE}")

    # --- Cleanup ---
    if os.path.exists(CHART_STRONG_SCALING): os.remove(CHART_STRONG_SCALING)
    if os.path.exists(CHART_WEAK_SCALING): os.remove(CHART_WEAK_SCALING)

def create_optimization_table(pdf, data):
    """Parses all results and builds a table comparing compiler optimizations."""
    if not data:
        pdf.cell(0, 10, "No data available to generate table.", ln=True)
        return
    profile_keys = list(data.keys())
    
    # Table Header
    pdf.set_font("Helvetica", "B", 10)
    num_profiles = len(profile_keys)
    col_width = 180 / (num_profiles + 1)
    header = ["Matrix Size"] + [k.replace("_", " ").title() for k in profile_keys]

    for item in header:
        pdf.cell(col_width, 10, item, border=1, align="C")
    pdf.ln()

    # Table Body
    pdf.set_font("Helvetica", size=9)
    try:
        first_profile_data = data.get(profile_keys[0], {})
        matrix_sizes = sorted(first_profile_data.get('general_perf', {}).get('hw2_b', {}).keys(), key=lambda x: int(x[1:]))
    except (KeyError, IndexError, TypeError):
        pdf.cell(0, 10, "No valid hw2-b performance data found to build table.", ln=True)
        return
        
    for size in matrix_sizes:
        row_data = []
        # Gather data for each profile for the current size
        for profile in profile_keys:
            best_gflops, best_threads = 0.0, 0
            try:
                schedules_data = data[profile]['general_perf']['hw2_b'][size]
                for schedule in schedules_data:
                    for thread, gflops in schedules_data[schedule].items():
                        # Robustly convert gflops to float, default to 0.0 on failure
                        try:
                            gflops_val = float(gflops)
                        except (ValueError, TypeError):
                            gflops_val = 0.0

                        if gflops_val > best_gflops:
                            best_gflops = gflops_val
                            best_threads = int(thread[1:])
            except (KeyError, TypeError):
                # This profile may not have data for this size
                pass
            row_data.append({'gflops': best_gflops, 'threads': best_threads})
        
        # Find the max Gflop/s in the row to highlight it
        max_gflops_in_row = max(d['gflops'] for d in row_data) if row_data else 0.0

        # Render the row
        pdf.cell(col_width, 10, size[1:] + "x" + size[1:], border=1, align="C")
        for cell_data in row_data:
            is_best = cell_data['gflops'] == max_gflops_in_row and max_gflops_in_row > 0.0
            
            if is_best:
                pdf.set_fill_color(200, 220, 255) # Light blue highlight
                fill = True
            else:
                fill = False

            cell_text = f"{cell_data['gflops']:.2f} ({cell_data['threads']}T)" if cell_data['gflops'] > 0 else "N/A"
            pdf.cell(col_width, 10, cell_text, border=1, align="C", fill=fill)
        pdf.ln()

def find_best_profile(data):
    """Finds the optimization profile that yielded the highest peak Gflop/s."""
    max_gflops = 0.0
    best_profile = None
    if not data: return "O3_default" # Return a default if no data
    
    for profile, profile_data in data.items():
        try:
            strong_scaling = profile_data.get('strong_scaling', {}).get('hw2_b_guided', {})
            if not strong_scaling: continue

            # Find the max gflops within this profile's strong scaling data
            current_max = 0.0
            for gflops_str in strong_scaling.values():
                try:
                    gflops_val = float(gflops_str)
                    if gflops_val > current_max:
                        current_max = gflops_val
                except (ValueError, TypeError):
                    continue

            if current_max > max_gflops:
                max_gflops = current_max
                best_profile = profile
        except (KeyError, TypeError):
            continue
    return best_profile if best_profile else list(data.keys())[0]

def generate_scaling_chart(data, title, xlabel, ylabel, filename, is_weak=False):
    """Generates and saves a single scaling chart."""
    if not data: return
    plt.figure(figsize=(10, 5.5))
    
    for label, results in data.items():
        try:
            threads_int = sorted([int(t[1:]) for t in results.keys()])
            threads_str = [f"T{t}" for t in threads_int]
            
            if is_weak:
                gflops = [float(results[t]['gflops']) for t in threads_str if results[t].get('gflops')]
                threads_plot = [int(t[1:]) for t in threads_str if results[t].get('gflops')]
            else:
                gflops = [float(results[t]) for t in threads_str if results.get(t)]
                threads_plot = [int(t[1:]) for t in threads_str if results.get(t)]
            
            if threads_plot: # Only plot if we have valid data points
                plt.plot(threads_plot, gflops, marker='o', linestyle='-', label=label.replace("_", " ").title())

        except (ValueError, TypeError, KeyError) as e:
            print(f"Warning: Could not plot data for '{label}' in '{title}' due to missing/invalid data. Error: {e}")
            continue

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plt.gca().has_data(): # Only add legend if something was plotted
        plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        print("Please run './run_benchmarks.sh' first to generate performance data.")
    else:
        with open(RESULTS_FILE, "r") as f:
            try:
                benchmark_results = json.load(f)
                generate_report(benchmark_results)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {RESULTS_FILE}. It may be empty or corrupted.")

