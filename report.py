import json
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import numpy as np
import os

# --- Configuration ---
RESULTS_FILENAME = "results.json"
PDF_FILENAME = "hw2.pdf"
LOG_FILENAME = "LOG.txt"

# --- Main Report Generation Logic ---
def generate_report(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    
    # Title
    pdf.cell(0, 10, "HW2 Performance Analysis Report", 0, 1, "C")
    pdf.ln(10)

    # --- Part 1: Performance Table for hw2-b ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. hw2-b: Gflop/s Performance with Different Schedules", 0, 1)
    pdf.set_font("Helvetica", "", 10)
    
    try:
        table_data = []
        sizes = data['general_perf']['hw2_b'].keys()
        
        for size_key in sorted(sizes, key=lambda s: int(s[1:])):
            size = int(size_key[1:])
            for sched_key, results in data['general_perf']['hw2_b'][size_key].items():
                schedule = sched_key.replace('schedule_', '')
                
                # Find the thread count with the best performance for this config
                best_t = 0
                max_gflops = 0.0
                for t_key, gflops_str in results.items():
                    try:
                        gflops = float(gflops_str)
                        t = int(t_key[1:])
                        if gflops > max_gflops:
                            max_gflops = gflops
                            best_t = t
                    except (ValueError, TypeError):
                        continue # Skip if gflops is 'error' or invalid
                
                if best_t > 0:
                    table_data.append([f"{size}x{size}", schedule.capitalize(), f"{max_gflops:.2f}", str(best_t)])
        
        if table_data:
            with pdf.table(col_widths=(40, 30, 40, 40), text_align="CENTER") as table:
                header = table.row()
                header.cell("Matrix Size")
                header.cell("Schedule")
                header.cell("Peak Performance (Gflop/s)")
                header.cell("Best Thread Count")
                for row_data in table_data:
                    row = table.row()
                    for cell_data in row_data:
                        row.cell(cell_data)
        else:
             pdf.cell(0, 10, "No valid performance data found for the table.", 0, 1, "I")

    except KeyError:
        pdf.multi_cell(0, 10, "Could not generate performance table. 'general_perf' data missing from results.json.", 0, "L")
    pdf.ln(10)

    # --- Part 2: Scaling Graphs ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "2. Strong and Weak Scaling Analysis", 0, 1)
    
    # Strong Scaling Plot
    try:
        strong_data = data['strong_scaling']['hw2_b_guided']
        threads = sorted([int(k[1:]) for k in strong_data.keys()])
        gflops_strong = [float(strong_data[f'T{t}']) for t in threads]
        speedup = [gflops / gflops_strong[0] for gflops in gflops_strong]

        plt.figure(figsize=(8, 4))
        plt.plot(threads, speedup, 'o-', label='Measured Speedup')
        plt.plot(threads, threads, 'r--', label='Ideal Linear Speedup')
        plt.title('Strong Scaling of hw2-b (guided)')
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        strong_scaling_chart = "strong_scaling.png"
        plt.savefig(strong_scaling_chart)
        pdf.image(strong_scaling_chart, x=None, y=None, w=150)
        os.remove(strong_scaling_chart)

    except (KeyError, IndexError, ValueError):
        pdf.multi_cell(0, 10, "Could not generate strong scaling plot. Data missing or invalid in results.json.", 0, "L")

    # Weak Scaling Plot
    try:
        weak_data = data['weak_scaling']['hw2_b_guided']
        threads_weak = sorted([int(k[1:]) for k in weak_data.keys()])
        gflops_weak = [float(weak_data[f'T{t}']['gflops']) for t in threads_weak]

        plt.figure(figsize=(8, 4))
        plt.plot(threads_weak, gflops_weak, 'o-')
        plt.title('Weak Scaling of hw2-b (guided)')
        plt.xlabel('Number of Threads')
        plt.ylabel('Performance (Gflop/s)')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.tight_layout()
        weak_scaling_chart = "weak_scaling.png"
        plt.savefig(weak_scaling_chart)
        pdf.image(weak_scaling_chart, x=None, y=None, w=150)
        os.remove(weak_scaling_chart)

    except (KeyError, ValueError):
        pdf.multi_cell(0, 10, "Could not generate weak scaling plot. Data missing or invalid in results.json.", 0, "L")

    pdf.ln(10)
    
    # --- Part 3: AI Tool Usage Reflection ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "3. Reflection on AI Tool Usage", 0, 1)
    pdf.set_font("Helvetica", "", 10)
    
    # Read from LOG.txt if it exists
    if os.path.exists(LOG_FILENAME):
        with open(LOG_FILENAME, 'r') as f:
            log_content = f.read()
        pdf.multi_cell(0, 5, log_content)
    else:
        pdf.multi_cell(0, 5, "LOG.txt not found. Please create this file with your reflections on using AI.")
        
    # Save the PDF
    pdf.output(PDF_FILENAME)
    print(f"Report successfully generated: {PDF_FILENAME}")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        with open(RESULTS_FILENAME, 'r') as f:
            results = json.load(f)
        generate_report(results)
    except FileNotFoundError:
        print(f"Error: {RESULTS_FILENAME} not found. Please run the benchmark script first.")
    except json.JSONDecodeError:
        print(f"Error: Could not parse {RESULTS_FILENAME}. It may be corrupted or empty.")

