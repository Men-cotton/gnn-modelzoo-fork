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
    # Line 3: [Throughput] Samples: 32843.58 samples/s (32843.54) | Edges: ...
    PYG_MULTI_THROUGHPUT = re.compile(r"\[Throughput\] Samples: ([\d\.]+) samples/s \(([\d\.]+)\)")

    # PyG Eval (New Format)
    # [Eval] Step=2000, Wall=77.8175s, Val_Acc=0.7004
    PYG_EVAL = re.compile(r"\[Eval\] Step=(\d+), Wall=([\d\.]+)s, Val_Acc=([\d\.]+)")

    # ModelZoo (WSE)
    TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    MZ_STEP = re.compile(r"GlobalStep=(\d+)")
    MZ_ACC = re.compile(r"eval/masked_accuracy = ([\d\.]+)")
    MZ_COMPUTE = re.compile(r"compute_time=([\d\.]+)")
    MZ_THROUGHPUT = re.compile(r"\| Train Device=.*Step=(\d+).*Rate=([\d\.]+) samples/sec, GlobalRate=([\d\.]+) samples/sec")
    MZ_COMPUTETIME_EXPLICIT = re.compile(r"ComputeTime Step=(\d+) compute_time=([\d\.]+)s")


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

        # ModelZoo Parsing State
        self._mz_start_time: Optional[datetime] = None
        self._mz_current_wall: float = 0.0
        self._mz_current_step: int = 0
        self._mz_step_to_compute: Dict[int, float] = {}
        self._mz_throughput_buffer: List[Dict] = []

    def parse(self) -> TrainingLogData:
        with open(self.filepath, "r") as f:
            for line in f:
                if self._parse_pyg_line(line):
                    continue
                self._parse_modelzoo_line(line)
        
        self._finalize_modelzoo_data()
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

        # 3. PyG Multi-Line Profile (Line 2)
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

        return False

    def _parse_modelzoo_line(self, line: str):
        # 1. Timestamp (Calc Wall Time)
        ts_match = LogPatterns.TIMESTAMP.search(line)
        if ts_match:
            try:
                dt = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if self._mz_start_time is None:
                    self._mz_start_time = dt
                self._mz_current_wall = (dt - self._mz_start_time).total_seconds()
            except ValueError:
                pass

        # 2. Step
        step_match = LogPatterns.MZ_STEP.search(line)
        if step_match:
            self._mz_current_step = int(step_match.group(1))

        # 3. Compute Time (Standard metric)
        comp_match = LogPatterns.MZ_COMPUTE.search(line)
        if comp_match:
            comp_val = float(comp_match.group(1))
            # Backfill compute time for the last eval entry if it was initialized to 0
            if self.data.eval_compute_times and self.data.eval_compute_times[-1] <= 0:
                self.data.eval_compute_times[-1] = comp_val

        # 4. Accuracy
        acc_match = LogPatterns.MZ_ACC.search(line)
        if acc_match:
            if self._mz_start_time is not None:
                self.data.eval_steps.append(self._mz_current_step)
                self.data.accuracies.append(float(acc_match.group(1)))
                self.data.eval_wall_times.append(self._mz_current_wall)
                self.data.eval_compute_times.append(0.0) # Will be backfilled

        # 5. Throughput
        tp_match = LogPatterns.MZ_THROUGHPUT.search(line)
        if tp_match and self._mz_start_time is not None:
            self._mz_throughput_buffer.append({
                "step": int(tp_match.group(1)),
                "wall": self._mz_current_wall,
                "local": float(tp_match.group(2)),
                "global": float(tp_match.group(3))
            })

        # 6. Explicit Compute Time
        ct_match = LogPatterns.MZ_COMPUTETIME_EXPLICIT.search(line)
        if ct_match:
            self._mz_step_to_compute[int(ct_match.group(1))] = float(ct_match.group(2))

    def _finalize_modelzoo_data(self):
        if not self._mz_throughput_buffer:
            return

        self._mz_throughput_buffer.sort(key=lambda x: x["step"])

        for entry in self._mz_throughput_buffer:
            step = entry["step"]
            comp_time = self._mz_step_to_compute.get(step, 0.0)
            
            self.data.train_steps.append(step)
            self.data.train_wall_times.append(entry["wall"])
            self.data.train_compute_times.append(comp_time)
            self.data.local_throughputs.append(entry["local"])
            self.data.global_throughputs.append(entry["global"])

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
        check_fn = lambda d: d.has_eval_data()
    elif "throughput" in metric_type:
        is_local = "local" in metric_type
        y_label = "Samples / Sec"
        title_base = "Local Rate" if is_local else "Global Rate"
        x_attrs = ("train_wall_times", "train_compute_times", "train_steps")
        y_attr = "local_throughputs" if is_local else "global_throughputs"
        check_fn = lambda d: d.has_train_data()
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
        # Determine average metrics for this run
        if len(data.step_fwds) > 0:
            # We have at least basic component data
            f = sum(data.step_fwds) / len(data.step_fwds)
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
                # Legacy fallback
                # Assuming Load = H2D roughly for old logs if no other info
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
        print("No breakdown data found for stacked bar chart.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = range(len(names))
    width = 0.6

    # Stack: Struc -> Fetch -> Fwd -> Bwd -> Opt -> CPU Overhead
    p1 = ax.bar(indices, strucs, width, label='H2D (Struc)')
    p2 = ax.bar(indices, fetchs, width, bottom=strucs, label='H2D (Fetch)')
    
    bottom_fwd = [s + fe for s, fe in zip(strucs, fetchs)]
    p3 = ax.bar(indices, fwds, width, bottom=bottom_fwd, label='Forward')
    
    bottom_bwd = [s + fe + f for s, fe, f in zip(strucs, fetchs, fwds)]
    p4 = ax.bar(indices, bwds, width, bottom=bottom_bwd, label='Backward')
    
    bottom_opt = [s + fe + f + b for s, fe, f, b in zip(strucs, fetchs, fwds, bwds)]
    p5 = ax.bar(indices, opts, width, bottom=bottom_opt, label='Optimizer')
    
    bottom_gap = [s + fe + f + b + o for s, fe, f, b, o in zip(strucs, fetchs, fwds, bwds, opts)]
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
    parser.add_argument("--output", default="result.png", help="Base output filename")
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

    # Synchronize Wall Times for Eval Logs
    sync_eval_wall_times(all_data)

    # Generate filenames
    base_name, ext = os.path.splitext(args.output)
    ext = ext if ext else ".png"
    
    # Create Plots
    plot_metric_set(all_data, "accuracy", f"{base_name}_accuracy{ext}")
    plot_metric_set(all_data, "local_throughput", f"{base_name}_throughput_local{ext}")
    plot_metric_set(all_data, "global_throughput", f"{base_name}_throughput_global{ext}")
    plot_throughput_breakdown(all_data, f"{base_name}_breakdown{ext}")

def sync_eval_wall_times(all_data: List[TrainingLogData]):
    """
    Replaces wall times in _eval logs with those from corresponding minimal logs.
    If no minimal log is found, clears eval_wall_times to exclude from Wall Time plot.
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
        if minimal_name in minimal_logs:
            target = minimal_logs[minimal_name]
            print(f"Aligning {d.name} with {target.name}...")
            
            # Map Step -> Wall Time from target
            step_map = {s: w for s, w in zip(target.train_steps, target.train_wall_times)}
            
            new_wall_times = []
            valid_indices = []
            
            for i, step in enumerate(d.eval_steps):
                if step in step_map:
                    new_wall_times.append(step_map[step])
                    valid_indices.append(i)
                else:
                    # If step mismatch, maybe interpolate? 
                    # For now, just keep original? No, user wants minimal log time.
                    # accessible. If exact step missing, we can't map accurately.
                    pass 

            # Update if we found matches
            if len(new_wall_times) == len(d.eval_steps):
                 d.eval_wall_times = new_wall_times
                 print(f"  -> Successfully replaced {len(new_wall_times)} timestamps.")
            else:
                 # Partial match?
                 # Case 1: Eval steps are subset of Train steps (Expected ideal)
                 # Case 2: Eval steps differ. 
                 # We will strict replace only if we can match. 
                 
                 # Actually, let's just replace assuming steps match index-wise? 
                 # Unsafe. Better to match by Step ID.
                 
                 # Refined approach:
                 final_walls = []
                 for step, old_wall in zip(d.eval_steps, d.eval_wall_times):
                     if step in step_map:
                         final_walls.append(step_map[step])
                     else:
                         # Fallback or keep? The request implies minimal log is the source of truth.
                         # usage "accuracy vs wall time ... refer to corresponding log".
                         # If step not in minimal log, we don't have a minimal wall time.
                         final_walls.append(old_wall) # Fallback to original?
                 
                 d.eval_wall_times = final_walls
                 print(f"  -> Replaced timestamps where possible.")

        else:
            # Standalone Eval Log
            print(f"Dropping Wall Time plot for {d.name} (no minimal log found).")
            d.eval_wall_times = [] # Clear to exclude from Wall Time plot

if __name__ == "__main__":
    main()
