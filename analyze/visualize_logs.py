import argparse
import re
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class TrainingLogData:
    """Class to hold training data extracted from logs"""
    name: str
    
    # Validation / Evaluation Metrics
    eval_steps: List[int] = field(default_factory=list)
    eval_wall_times: List[float] = field(default_factory=list)
    eval_compute_times: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)

    # Training Throughput Metrics
    train_steps: List[int] = field(default_factory=list)
    train_wall_times: List[float] = field(default_factory=list)
    train_compute_times: List[float] = field(default_factory=list)
    local_throughputs: List[float] = field(default_factory=list)
    global_throughputs: List[float] = field(default_factory=list)

    # Detailed Phase Metrics (Averages from Summary)
    avg_load: float = 0.0
    avg_fwd: float = 0.0
    avg_bwd: float = 0.0
    avg_opt: float = 0.0
    
    # Per-step breakdown
    step_loads: List[float] = field(default_factory=list)
    # Granular H2D (if available)
    step_preps: List[float] = field(default_factory=list)
    step_h2d_struc: List[float] = field(default_factory=list) 
    step_h2d_fetch: List[float] = field(default_factory=list)
    step_h2ds: List[float] = field(default_factory=list) # Aggregate if granular not available
    
    step_fwds: List[float] = field(default_factory=list)
    step_bwds: List[float] = field(default_factory=list)
    step_opts: List[float] = field(default_factory=list)

    def has_eval_data(self) -> bool:
        return len(self.eval_steps) > 0

    def has_train_data(self) -> bool:
        return len(self.train_steps) > 0

# -----------------------------------------------------------------------------
# Parser Logic
# -----------------------------------------------------------------------------

class LogPatterns:
    """Regex patterns for parsing logs"""
    # PyG Native (Multi-Line New Format)
    # Line 1: [Step=0080] Wall=11.4164s | Loss=1.8993
    PYG_MULTI_STEP = re.compile(r"\[Step=(\d+)\] Wall=([\d\.]+)s \| Loss=([\d\.]+)")
    # Line 2: [Profile] Avg ms/step | Load: 23.309 (Prep: 10.757, Struc: 0.568, Fetch: 11.984) | Fwd: 5.622 | Bwd: 5.306 | Opt: 0.152 | GPU_Tot: 23.631
    PYG_MULTI_PROFILE = re.compile(r"\[Profile\] Avg ms/step \| Load: ([\d\.]+) \(Prep: ([\d\.]+), Struc: ([\d\.]+), Fetch: ([\d\.]+)\) \| Fwd: ([\d\.]+) \| Bwd: ([\d\.]+) \| Opt: ([\d\.]+) \| GPU_Tot: ([\d\.]+)")
    # Line 2 (CSZoo variant): [Profile] Avg ms/step | Load: 1705.805 | Host_Submit(Fwd: 0.000, Bwd: 0.000, Opt: 0.000) | Residual(Dev): 0.458 | Iter_Wall: 0.458
    CSZOO_PROFILE = re.compile(r"\[Profile\] Avg ms/step \| Load: ([\d\.]+) \| Host_Submit\(Fwd: ([\d\.]+), Bwd: ([\d\.]+), Opt: ([\d\.]+)\) \| Residual\(Dev\): ([\d\.]+) \| Iter_Wall: ([\d\.]+)")
    
    # Line 3: [Throughput] Samples: 32843.58 samples/s (32843.54) | Edges: ...
    PYG_MULTI_THROUGHPUT = re.compile(r"\[Throughput\] Samples: ([\d\.]+) samples/s \(([\d\.]+)\)")

    # PyG Eval (New Format)
    # [Eval] Step=2000, Wall=77.8175s, Val_Acc=0.7004
    PYG_EVAL = re.compile(r"\[Eval\] Step=(\d+), Wall=([\d\.]+)s, Val_Acc=([\d\.]+)")

    # WSE Eval (Header)
    # | Eval Device=CSX, GlobalStep=20, Batch=8, ...
    WSE_EVAL_HEADER = re.compile(r"\| Eval Device=CSX, GlobalStep=(\d+),")

    # WSE Eval (Metric)
    #     - eval/masked_accuracy = 0.3185677230358124
    WSE_EVAL_METRIC = re.compile(r"\s+-\s+eval/masked_accuracy\s+=\s+([\d\.]+)")

class LogParser:
    """Parses log files into TrainingLogData"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.data = TrainingLogData(name=self.filename)
        
        # Multi-line Parsing State
        self._current_step_data = {}
        self._current_compute_time = 0.0
        self._last_step = 0
        self._current_wse_eval_step = None

    def parse(self) -> TrainingLogData:
        with open(self.filepath, "r") as f:
            for line in f:
                self._parse_pyg_line(line)
        return self.data

    def _parse_pyg_line(self, line: str) -> bool:
        # 1. PyG Eval
        match = LogPatterns.PYG_EVAL.search(line)
        if match:
            self.data.eval_steps.append(int(match.group(1)))
            self.data.eval_wall_times.append(float(match.group(2)))
            self.data.accuracies.append(float(match.group(3)))
            self.data.eval_compute_times.append(0.0) 
            return True

        # 2. PyG Multi-Line Step (Line 1)
        match = LogPatterns.PYG_MULTI_STEP.search(line)
        if match:
            self._current_step_data = {
                "step": int(match.group(1)),
                "wall": float(match.group(2))
            }
            return True

        # 3a. PyG Multi-Line Profile (Standard)
        match = LogPatterns.PYG_MULTI_PROFILE.search(line)
        if match:
            if self._current_step_data:
                # ms to sec
                self._current_step_data["load"] = float(match.group(1)) / 1000.0
                self._current_step_data["prep"] = float(match.group(2)) / 1000.0
                self._current_step_data["struc"] = float(match.group(3)) / 1000.0
                self._current_step_data["fetch"] = float(match.group(4)) / 1000.0
                self._current_step_data["fwd"] = float(match.group(5)) / 1000.0
                self._current_step_data["bwd"] = float(match.group(6)) / 1000.0
                self._current_step_data["opt"] = float(match.group(7)) / 1000.0
                self._current_step_data["gpu_tot"] = float(match.group(8)) / 1000.0
            return True

        # 3b. CSZoo Profile
        match = LogPatterns.CSZOO_PROFILE.search(line)
        if match:
            if self._current_step_data:
                # ms to sec
                # Load: (1) | Host_Submit(Fwd: (2), Bwd: (3), Opt: (4)) | Residual: (5) | Iter_Wall: (6)
                self._current_step_data["load"] = float(match.group(1)) / 1000.0
                self._current_step_data["prep"] = 0.0
                self._current_step_data["struc"] = 0.0
                self._current_step_data["fetch"] = 0.0
                
                # These are "Host_Submit" times which map to Fwd/Bwd/Opt for breakdown purposes if > 0
                self._current_step_data["fwd"] = float(match.group(2)) / 1000.0
                self._current_step_data["bwd"] = float(match.group(3)) / 1000.0
                self._current_step_data["opt"] = float(match.group(4)) / 1000.0
                
                # Use Iter_Wall as total GPU time (compute time)
                self._current_step_data["gpu_tot"] = float(match.group(6)) / 1000.0
            return True

        # 4. PyG Multi-Line Throughput (Line 3) - Trigger flush
        match = LogPatterns.PYG_MULTI_THROUGHPUT.search(line)
        if match:
            if self._current_step_data and "step" in self._current_step_data:
                # Flush
                d = self._current_step_data
                current_step = d["step"]
                
                # "gpu_tot" is avg time per step for the interval
                # If step jumps from 100 to 200, we have 100 steps
                delta_steps = current_step - self._last_step
                if delta_steps > 0:
                    avg_gpu_time = d.get("gpu_tot", 0.0)
                    self._current_compute_time += avg_gpu_time * delta_steps
                
                self._last_step = current_step

                self.data.train_steps.append(current_step)
                self.data.train_wall_times.append(d["wall"])
                self.data.step_loads.append(d.get("load", 0.0))
                self.data.step_preps.append(d.get("prep", 0.0))
                self.data.step_h2d_struc.append(d.get("struc", 0.0))
                self.data.step_h2d_fetch.append(d.get("fetch", 0.0))
                self.data.step_fwds.append(d.get("fwd", 0.0))
                self.data.step_bwds.append(d.get("bwd", 0.0))
                self.data.step_opts.append(d.get("opt", 0.0))
                
                self.data.train_compute_times.append(self._current_compute_time)
                
                self.data.global_throughputs.append(float(match.group(1)))
                self.data.local_throughputs.append(float(match.group(2)))
                
                self._current_step_data = {}
            return True

        # 5. WSE Eval Header
        match = LogPatterns.WSE_EVAL_HEADER.search(line)
        if match:
            self._current_wse_eval_step = int(match.group(1))
            return True

        # 6. WSE Eval Metric
        match = LogPatterns.WSE_EVAL_METRIC.search(line)
        if match and self._current_wse_eval_step is not None:
            self.data.eval_steps.append(self._current_wse_eval_step)
            # Wall/Compute time unknown here, set to 0.0 for sync later
            self.data.eval_wall_times.append(0.0) 
            self.data.eval_compute_times.append(0.0)
            self.data.accuracies.append(float(match.group(1)))
            self._current_wse_eval_step = None
            return True

        return False

# -----------------------------------------------------------------------------
# Visualization Logic
# -----------------------------------------------------------------------------

def plot_metric_set(all_data: List[TrainingLogData], 
                    metric_type: str, 
                    output_file: str):
    """
    Creates 3 subplots (Wall Time, Compute Time, Steps).
    metric_type: 'accuracy', 'local_throughput', 'global_throughput'
    """
    
    if metric_type == "accuracy":
        y_label = "Validation Accuracy"
        title_base = "Accuracy"
        x_attrs = ("eval_wall_times", "eval_compute_times", "eval_steps")
        y_attr = "accuracies"
        # Filter out single-point evaluations (e.g. final eval only) as they don't show convergence
        check_fn = lambda d: d.has_eval_data() and len(d.eval_steps) > 1
    elif "throughput" in metric_type:
        is_local = "local" in metric_type
        y_label = "Samples / Sec"
        title_base = "Local Rate" if is_local else "Global Rate"
        x_attrs = ("train_wall_times", "train_compute_times", "train_steps")
        y_attr = "local_throughputs" if is_local else "global_throughputs"
        check_fn = lambda d: d.has_train_data() and "_eval" not in d.name
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    valid_data = [d for d in all_data if check_fn(d)]
    if not valid_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [f"{title_base} vs Wall Time", f"{title_base} vs Compute Time", f"{title_base} vs Steps"]
    x_labels = ["Wall Time (s)", "Compute Time (s)", "Steps"]

    for i, ax in enumerate(axes):
        x_attr = x_attrs[i]
        
        for data in valid_data:
            x_vals = getattr(data, x_attr)
            y_vals = getattr(data, y_attr)
            
            if i == 1: # Compute Time
                filtered = [(x, y) for x, y in zip(x_vals, y_vals) if x > 0]
                if filtered:
                    x_vals, y_vals = zip(*filtered)
                else:
                    x_vals, y_vals = [], []

            if x_vals and y_vals:
                ax.plot(x_vals, y_vals, marker='.' if metric_type == "accuracy" else None, label=data.name)

        # Set Axis Limits
        if i in [0, 1, 2]: # Wall Time, Compute Time, Steps
            ax.set_xlim(left=0)
        
        if "throughput" in metric_type:
            ax.set_ylim(bottom=0)

        ax.set_xlabel(x_labels[i])
        ax.set_ylabel(y_label)
        ax.set_title(titles[i])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close(fig)

def plot_throughput_breakdown(all_data: List[TrainingLogData], output_file: str):
    """
    Creates a stacked bar chart showing the breakdown of time per step:
    GPU Components: H2D_Struc, H2D_Fetch, Fwd, Bwd, Opt
    Gap: Wall - GPU_Total (Labelled as CPU Overhead / Idle)
    """
    if not all_data:
        return

    names = []
    
    # GPU Components
    strucs = []
    fetchs = []
    fwds = []
    bwds = []
    opts = []
    
    # Gap
    cpu_overhead_idles = []

    has_data = False

    for data in all_data:
        # Skip Eval Logs for Breakdown
        if "_eval" in data.name:
            continue

        # Determine average metrics for this run
        if len(data.step_fwds) > 0:
            # Check if breakdown is meaningful (non-negligible fwd/bwd)
            avg_fwd_time = sum(data.step_fwds) / len(data.step_fwds)
            # Threshold: 1 microsecond. If less, likely WSE run with async host submission or empty breakdown.
            if avg_fwd_time < 1e-6:
                print(f"Skipping breakdown for {data.name} (negligible forward time, likely WSE run).")
                continue

            # We have at least basic component data
            f = avg_fwd_time
            b = sum(data.step_bwds) / len(data.step_bwds)
            o = sum(data.step_opts) / len(data.step_opts)
            
            # H2D Breakdown
            if len(data.step_h2d_struc) > 0 and len(data.step_h2d_fetch) > 0:
                s = sum(data.step_h2d_struc) / len(data.step_h2d_struc)
                fe = sum(data.step_h2d_fetch) / len(data.step_h2d_fetch)
            elif len(data.step_h2ds) > 0:
                # Fallback to aggregated H2D
                s = sum(data.step_h2ds) / len(data.step_h2ds)
                fe = 0.0 # Assign all to structure/generic
            elif len(data.step_loads) > 0:
                # Legacy fallback / New CSZoo Load
                # In CSZoo logs Load is separate, but we can treat it as Structure/Prep for viz if desired,
                # or just use it as 'H2D' block. For now, map simple Load to Struct if others are empty.
                # Actually, standard PyG breakdown usually has separate H2D.
                # If we only have 'Load' and others are 0, we can map it to 'Struc' to be visible.
                s = sum(data.step_loads) / len(data.step_loads)
                fe = 0.0
            else:
                s, fe = 0.0, 0.0

            # Calculate Wall Time per Step
            if len(data.train_wall_times) > 1:
                duration = data.train_wall_times[-1] - data.train_wall_times[0]
                steps = data.train_steps[-1] - data.train_steps[0]
                w = duration / steps if steps > 0 else 0.0
            else:
                w = 0.0
                
        else:
            continue

        # Calculate Gap
        gpu_total = s + fe + f + b + o
        if w > 0:
            gap = max(0.0, w - gpu_total)
        else:
            gap = 0.0

        names.append(data.name)
        strucs.append(s)
        fetchs.append(fe)
        fwds.append(f)
        bwds.append(b)
        opts.append(o)
        cpu_overhead_idles.append(gap)
        has_data = True

    if not has_data:
        print("No valid breakdown data found for stacked bar chart.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = range(len(names))
    width = 0.6

    # Stack: Fwd -> Bwd -> Opt -> Struc -> Fetch -> CPU Overhead
    p1 = ax.bar(indices, fwds, width, label='Forward')

    bottom_bwd = fwds
    p2 = ax.bar(indices, bwds, width, bottom=bottom_bwd, label='Backward')

    bottom_opt = [f + b for f, b in zip(fwds, bwds)]
    p3 = ax.bar(indices, opts, width, bottom=bottom_opt, label='Optimizer')

    bottom_struc = [f + b + o for f, b, o in zip(fwds, bwds, opts)]
    p4 = ax.bar(indices, strucs, width, bottom=bottom_struc, label='H2D (Struc/Load)')

    bottom_fetch = [f + b + o + s for f, b, o, s in zip(fwds, bwds, opts, strucs)]
    p5 = ax.bar(indices, fetchs, width, bottom=bottom_fetch, label='H2D (Fetch)')
    
    bottom_gap = [f + b + o + s + fe for f, b, o, s, fe in zip(fwds, bwds, opts, strucs, fetchs)]
    p6 = ax.bar(indices, cpu_overhead_idles, width, bottom=bottom_gap, label='CPU Overhead / Idle', hatch='//')

    ax.set_ylabel('Time per Step (s)')
    ax.set_title('Training Step Time Breakdown')
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Breakdown plot saved to {output_file}")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize Time-to-Accuracy and Throughput from training logs")
    parser.add_argument("log_dir", help="Path to the directory containing log files")
    parser.add_argument("--output", default="result.svg", help="Base output filename")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"Error: {args.log_dir} is not a directory.")
        return

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
    if not log_files:
        print(f"No .log files found in {args.log_dir}")
        return

    print(f"Found {len(log_files)} log files.")

    all_data: List[TrainingLogData] = []
    
    for f in log_files:
        print(f"Parsing {os.path.basename(f)}...")
        log_parser = LogParser(f)
        data = log_parser.parse()
        
        if data.has_eval_data() or data.has_train_data():
            all_data.append(data)
        else:
            print(f"Warning: No valid metrics found in {data.name}")

    if not all_data:
        print("No plotable data found.")
        return

    # Synchronize Wall Times and Compute Times for Eval Logs
    sync_eval_metrics(all_data)

    # Generate filenames
    base_name, ext = os.path.splitext(args.output)
    ext = ext if ext else ".svg"
    
    # Create Plots
    plot_metric_set(all_data, "accuracy", f"{base_name}_accuracy{ext}")
    plot_metric_set(all_data, "local_throughput", f"{base_name}_throughput_local{ext}")
    plot_metric_set(all_data, "global_throughput", f"{base_name}_throughput_global{ext}")
    plot_throughput_breakdown(all_data, f"{base_name}_breakdown{ext}")
    plot_metrics_table(all_data, f"{base_name}_metrics_table{ext}")

def plot_metrics_table(all_data: List[TrainingLogData], output_file: str):
    """
    Computes and plots a table of metrics:
    - Last Global Rate * Total Wall Time
    - Last Global Rate * Total Compute Time
    """
    table_data = []
    headers = ["Log Name", "Average Rate * Wall", "Average Rate * Compute"]

    for data in all_data:
        # Only consider logs with valid training data
        if not data.has_train_data():
            continue
            
        # Skip purely eval logs if they don't have their own training data (already handled by check above)
        if "_eval" in data.name and not data.train_steps:
             continue

        last_global_rate = data.global_throughputs[-1] if data.global_throughputs else 0.0
        total_wall_time = data.train_wall_times[-1] if data.train_wall_times else 0.0
        total_compute_time = data.train_compute_times[-1] if data.train_compute_times else 0.0

        # Compute Metrics (Total Samples Processed Estimate?)
        # User requested: Last Global Rate * Time
        metric_wall = last_global_rate * total_wall_time
        metric_compute = last_global_rate * total_compute_time

        table_data.append([
            data.name,
            f"{metric_wall:,.0f}",
            f"{metric_compute:,.0f}"
        ])

    if not table_data:
        print("No data for metrics table.")
        return

    # Sort by Log Name
    table_data.sort(key=lambda x: x[0])

    # Plot Table
    fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.5 + 1))
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Metrics table saved to {output_file}")
    plt.close(fig)

def sync_eval_metrics(all_data: List[TrainingLogData]):
    """
    Replaces wall times and compute times in _eval logs with those from corresponding minimal logs.
    If no minimal log is found, clears eval_wall_times/eval_compute_times to exclude from plots.
    """
    # Create a lookup for minimal logs
    minimal_logs = {}
    for d in all_data:
        if not d.name.endswith("_eval.log") and "_eval" not in d.name:
            minimal_logs[d.name] = d

    for d in all_data:
        # Check if it's an eval log
        is_eval = d.name.endswith("_eval.log") or "_eval" in d.name
        if not is_eval:
            continue

        # Try to find counterpart
        # Simple heuristic: remove "_eval"
        minimal_name = d.name.replace("_eval", "")
        target = None
        if minimal_name in minimal_logs:
            target = minimal_logs[minimal_name]
        elif d.has_train_data():
            # Fallback: Sync with self if training data exists in the SAME log
            target = d
            
        if target:
            if target is d:
                print(f"Aligning {d.name} with self (contains both train and eval)...")
            else:
                print(f"Aligning {d.name} with {target.name}...")
            
            # Map Step -> Wall Time & Compute Time from target
            step_to_wall = {s: w for s, w in zip(target.train_steps, target.train_wall_times)}
            step_to_compute = {s: c for s, c in zip(target.train_steps, target.train_compute_times)}
            
            new_wall_times = []
            new_compute_times = []
            
            for i, step in enumerate(d.eval_steps):
                # Wall Time
                if step in step_to_wall:
                    new_wall_times.append(step_to_wall[step])
                else:
                    new_wall_times.append(d.eval_wall_times[i]) # Keep original (0.0 or valid)

                # Compute Time
                if step in step_to_compute:
                    new_compute_times.append(step_to_compute[step])
                else:
                    new_compute_times.append(d.eval_compute_times[i]) # Keep original
            
            d.eval_wall_times = new_wall_times
            d.eval_compute_times = new_compute_times
            print(f"  -> Replaced timestamps/compute times where possible.")

        else:
            # Standalone Eval Log
            print(f"Dropping Wall/Compute Time plot for {d.name} (no minimal log found).")
            d.eval_wall_times = [] 
            d.eval_compute_times = []

if __name__ == "__main__":
    main()
