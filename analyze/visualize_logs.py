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
    """ログから抽出された学習データを保持するクラス"""
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

    def has_eval_data(self) -> bool:
        return len(self.eval_steps) > 0

    def has_train_data(self) -> bool:
        return len(self.train_steps) > 0

# -----------------------------------------------------------------------------
# Parser Logic
# -----------------------------------------------------------------------------

class LogPatterns:
    """ログ解析に使用する正規表現パターン定義"""
    # PyG Native
    PYG_EVAL = re.compile(r"\[eval @ step (\d+)\] val_acc=([\d\.]+) wall=([\d\.]+)s compute=([\d\.]+)s")
    PYG_TRAIN = re.compile(r"\[step (\d+)\] .* wall=([\d\.]+)s compute=([\d\.]+)s Rate=([\d\.]+) samples/sec GlobalRate=([\d\.]+) samples/sec")

    # ModelZoo (WSE)
    TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    MZ_STEP = re.compile(r"GlobalStep=(\d+)")
    MZ_ACC = re.compile(r"eval/masked_accuracy = ([\d\.]+)")
    MZ_COMPUTE = re.compile(r"compute_time=([\d\.]+)")
    MZ_THROUGHPUT = re.compile(r"\| Train Device=.*Step=(\d+).*Rate=([\d\.]+) samples/sec, GlobalRate=([\d\.]+) samples/sec")
    MZ_COMPUTETIME_EXPLICIT = re.compile(r"ComputeTime Step=(\d+) compute_time=([\d\.]+)s")


class LogParser:
    """ログファイルを解析し、TrainingLogDataを生成するクラス"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.data = TrainingLogData(name=self.filename)
        
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
            self.data.accuracies.append(float(match.group(2)))
            self.data.eval_wall_times.append(float(match.group(3)))
            self.data.eval_compute_times.append(float(match.group(4)))
            return True

        # 2. PyG Training
        match = LogPatterns.PYG_TRAIN.search(line)
        if match:
            self.data.train_steps.append(int(match.group(1)))
            self.data.train_wall_times.append(float(match.group(2)))
            self.data.train_compute_times.append(float(match.group(3)))
            self.data.local_throughputs.append(float(match.group(4)))
            self.data.global_throughputs.append(float(match.group(5)))
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
        """ModelZooの一時データを統合・ソートしてメインデータ構造に格納"""
        if not self._mz_throughput_buffer:
            return

        # Sort buffer by step
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
    3つのサブプロット（Wall Time, Compute Time, Steps）を作成する汎用関数。
    metric_type: 'accuracy', 'local_throughput', 'global_throughput'
    """
    
    # 設定の定義
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

    # データがあるか確認
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
            
            # Compute Timeプロット(index 1)の場合、0を除外してプロットする
            if i == 1:
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

    # Generate filenames
    base_name, ext = os.path.splitext(args.output)
    ext = ext if ext else ".png"
    
    # Create Plots
    plot_metric_set(all_data, "accuracy", f"{base_name}_accuracy{ext}")
    plot_metric_set(all_data, "local_throughput", f"{base_name}_throughput_local{ext}")
    plot_metric_set(all_data, "global_throughput", f"{base_name}_throughput_global{ext}")

if __name__ == "__main__":
    main()
