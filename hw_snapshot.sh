#!/usr/bin/env bash
# hw_snapshot.sh
# Run: lscpu -> nvidia-smi -> nvidia-smi topo -m -> numactl -H (in order)
# Logs: per-command files + combined log

set -u
set -o pipefail

ts="$(date +%Y%m%d_%H%M%S)"
host="$(hostname 2>/dev/null || echo unknown-host)"
outdir="${1:-hwinfo_${host}_${ts}}"

mkdir -p "$outdir"

combined="${outdir}/all_${host}_${ts}.log"
summary="${outdir}/summary_${host}_${ts}.txt"
meta="${outdir}/meta_${host}_${ts}.txt"

command_exists() { command -v "$1" >/dev/null 2>&1; }

write_meta() {
  {
    echo "timestamp: $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "hostname : $host"
    echo "user     : $(id -un 2>/dev/null || whoami)"
    echo "pwd      : $(pwd)"
    echo "uname    : $(uname -a 2>/dev/null || true)"
    echo "SLURM_JOB_ID      : ${SLURM_JOB_ID:-}"
    echo "SLURM_NODELIST    : ${SLURM_NODELIST:-}"
    echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-}"
    echo
  } | tee -a "$combined" > "$meta"
}

run_cmd() {
  local label="$1"; shift
  local outfile="$1"; shift

  {
    echo "============================================================"
    echo "## ${label}"
    echo "## time: $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "## cmd : $*"
    echo "============================================================"
  } | tee -a "$combined" > "$outfile"

  if ! command_exists "$1"; then
    echo "SKIP: command not found: $1" | tee -a "$combined" >> "$outfile"
    echo "SKIP: ${label} (command not found: $1)" >> "$summary"
    echo >> "$combined"
    return 0
  fi

  # Run and append stdout/stderr to both per-command file and combined log
  if "$@" 2>&1 | tee -a "$combined" >> "$outfile"; then
    echo "OK  : ${label}" >> "$summary"
  else
    rc=$?
    echo "FAIL: ${label} (exit=$rc)" | tee -a "$combined" >> "$outfile"
    echo "FAIL: ${label} (exit=$rc)" >> "$summary"
    echo >> "$combined"
    return $rc
  fi

  echo >> "$combined"
  return 0
}

write_meta

run_cmd "lscpu"                "${outdir}/01_lscpu_${host}_${ts}.log"                lscpu
run_cmd "nvidia-smi"           "${outdir}/02_nvidia-smi_${host}_${ts}.log"           nvidia-smi
run_cmd "nvidia-smi topo -m"   "${outdir}/03_nvidia-smi_topo-m_${host}_${ts}.log"    nvidia-smi topo -m
run_cmd "numactl -H"           "${outdir}/04_numactl_-H_${host}_${ts}.log"           numactl -H

echo "Done. Logs saved under: ${outdir}"
echo "Combined: ${combined}"
echo "Summary : ${summary}"
