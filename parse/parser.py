import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_slurm_logs(log_dir="logs"):
    print(f"Scanning directory: {log_dir}/ for .out files...")
    
    # Regex patterns based on our bash script
    worker_pattern = re.compile(r"Workers \(MPI Ranks\):\s*(\d+)")
    rec_pattern = re.compile(r"RUNTIME_RECONSTRUCT_(\d+)_([0-9\.]+)=([0-9\.]+)")
    vec_pattern = re.compile(r"RUNTIME_VECTOR_(\d+)_([0-9\.]+)=([0-9\.]+)")
    
    parsed_data = []
    
    # Find all out files in the logs directory
    log_files = glob.glob(os.path.join(log_dir, "*.out"))
    
    if not log_files:
        print("No log files found! Make sure your logs/ directory exists and contains .out files.")
        return pd.DataFrame()

    for filepath in log_files:
        with open(filepath, 'r') as f:
            workers = None
            last_rec_time = None
            last_size = None
            last_density = None
            
            for line in f:
                # 1. Find the worker count for this file
                w_match = worker_pattern.search(line)
                if w_match:
                    workers = int(w_match.group(1))
                    continue
                
                # 2. Find reconstruct time
                rec_match = rec_pattern.search(line)
                if rec_match and workers is not None:
                    last_size = int(rec_match.group(1))
                    last_density = float(rec_match.group(2))
                    last_rec_time = float(rec_match.group(3))
                    continue
                
                # 3. Find vector time and pair it with the reconstruct time
                vec_match = vec_pattern.search(line)
                if vec_match and workers is not None and last_rec_time is not None:
                    v_size = int(vec_match.group(1))
                    v_density = float(vec_match.group(2))
                    v_time = float(vec_match.group(3))
                    
                    # Ensure they match the same config iteration
                    if v_size == last_size and v_density == last_density:
                        total_runtime = last_rec_time + v_time
                        parsed_data.append({
                            'workers': workers,
                            'size': v_size,
                            'density': v_density,
                            'runtime': total_runtime
                        })
                    
                    # Reset for the next iteration
                    last_rec_time = None

    df = pd.DataFrame(parsed_data)
    print(f"Successfully parsed {len(df)} completed tests.")
    return df

def plot_scaling_results(df):
    if df.empty:
        return

    # Calculate the mean runtime for each configuration
    summary_df = df.groupby(['size', 'density', 'workers']).agg(
        mean_runtime=('runtime', 'mean'),
        completed_tests=('runtime', 'count')
    ).reset_index()

    # Get unique sizes and densities and sort them so the grid looks logical
    sizes = sorted(summary_df['size'].unique())
    densities = sorted(summary_df['density'].unique())
    
    fig, axes = plt.subplots(len(sizes), len(densities), figsize=(15, 12), sharex=True)
    fig.suptitle('Mean Total Runtime vs Worker Count', fontsize=18, fontweight='bold', y=0.98)
    
    # Loop through sizes (rows) and densities (columns)
    for row_idx, sz in enumerate(sizes):
        for col_idx, den in enumerate(densities):
            ax = axes[row_idx, col_idx] if len(sizes) > 1 else axes[col_idx]
            
            # Filter data for this subplot
            plot_data = summary_df[(summary_df['size'] == sz) & (summary_df['density'] == den)]
            plot_data = plot_data.sort_values('workers')
            
            # Plot
            ax.bar(plot_data['workers'], plot_data['mean_runtime'], color='#4C72B0', edgecolor='black')
            
            # Formatting
            ax.set_title(f'N = {sz} | Density = {den}', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(range(1, 11))
            
            if col_idx == 0:
                ax.set_ylabel('Mean Runtime (s)')
            if row_idx == len(sizes) - 1:
                ax.set_xlabel('MPI Workers')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure to a file so you can easily download it from the HPC
    output_filename = "scaling_results_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved successfully as '{output_filename}'")
    
    # Also attempt to show it (will only work if you have X11 forwarding or are running locally)
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    df = parse_slurm_logs("logs")
    plot_scaling_results(df)