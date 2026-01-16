import argparse
import re
import matplotlib.pyplot as plt
import os
import glob

def parse_logs(log_file):
    steps = []
    wall_times = []
    compute_times = []
    accuracies = []

    # PyG Native Run (GPU) log pattern
    # Example: [eval @ step 92] val_acc=0.7680 wall=24.72s compute=19.45s
    pyg_gpu_pattern = re.compile(r"\[eval @ step (\d+)\] val_acc=([\d\.]+) wall=([\d\.]+)s compute=([\d\.]+)s")
    
    # ModelZoo Run (WSE) log patterns
    timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    step_pattern = re.compile(r"GlobalStep=(\d+)")
    acc_pattern = re.compile(r"eval/masked_accuracy = ([\d\.]+)")
    compute_pattern = re.compile(r"compute_time=([\d\.]+)")

    # State for ModelZoo parsing
    start_time = None
    current_compute = 0.0
    current_step = 0
    
    from datetime import datetime

    with open(log_file, "r") as f:
        for line in f:
            # Check for PyG (GPU) format first
            match = pyg_gpu_pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                accuracies.append(float(match.group(2)))
                wall_times.append(float(match.group(3)))
                compute_times.append(float(match.group(4)))
                continue

            # ModelZoo (WSE) Parsing
            # 1. Timestamp for Wall Time
            ts_match = timestamp_pattern.search(line)
            current_wall = 0.0
            if ts_match:
                dt = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if start_time is None:
                    start_time = dt
                current_wall = (dt - start_time).total_seconds()
            
            # 2. Step
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))

            # 3. Compute Time
            comp_match = compute_pattern.search(line)
            if comp_match:
                current_compute = float(comp_match.group(1))
                # Backfill if compute time is logged after accuracy (e.g. at validation end)
                if steps and compute_times[-1] <= 0:
                     compute_times[-1] = current_compute

            # 4. Accuracy
            acc_match = acc_pattern.search(line)
            if acc_match:
                acc = float(acc_match.group(1))
                # Only record if we have a valid timestamp (which we should for ModelZoo logs)
                if start_time is not None:
                    steps.append(current_step)
                    accuracies.append(acc)
                    wall_times.append(current_wall)
                    # Use accumulated compute time. 
                    # Initialize with 0.0 (or current if it came before).
                    # If it executes AFTER, the next loop iteration will catch logic #3 and backfill.
                    compute_times.append(0.0)

    # If ModelZoo mode and compute_times are all 0, maybe warn or set to Wall Time for visualization?
    # User requested "Compute Time" log. If missing, line will be flat at 0.
    
    return steps, wall_times, compute_times, accuracies

def main():
    parser = argparse.ArgumentParser(description="Visualize Time-to-Accuracy from training logs")
    parser.add_argument("log_dir", help="Path to the directory containing log files")
    parser.add_argument("--output", default="time_to_accuracy.png", help="Output image filename")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"Error: {args.log_dir} is not a directory.")
        return

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
    if not log_files:
        print(f"No .log files found in {args.log_dir}")
        return

    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f" - {os.path.basename(f)}")

    # Store data for plotting
    all_data = []
    
    for log_file in log_files:
        steps, wall_times, compute_times, accuracies = parse_logs(log_file)
        if steps:
            all_data.append({
                "name": os.path.basename(log_file),
                "steps": steps,
                "wall_times": wall_times,
                "compute_times": compute_times,
                "accuracies": accuracies
            })
        else:
            print(f"Warning: No metrics found in {os.path.basename(log_file)}")

    if not all_data:
        print("No plotable data found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # colors = plt.cm.tab10.colors  # Use a colormap
    
    for i, data in enumerate(all_data):
        label = data["name"]
        
        # 1. Accuracy vs Wall Time
        # axes[0].plot(data["wall_times"], data["accuracies"], marker='o', label=label)
        axes[0].plot(data["wall_times"], data["accuracies"], marker='.', label=label)

        # 2. Accuracy vs Compute Time
        axes[1].plot(data["compute_times"], data["accuracies"], marker='.', label=label)

        # 3. Accuracy vs Steps
        axes[2].plot(data["steps"], data["accuracies"], marker='.', label=label)

    # Configure axes
    axes[0].set_xlabel("Wall Time (s)")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Accuracy vs Wall Time")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("Compute Time (s)")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Accuracy vs Compute Time")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Validation Accuracy")
    axes[2].set_title("Accuracy vs Steps")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
