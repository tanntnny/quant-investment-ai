#!/usr/bin/env bash

#SBATCH --job-name=train-qai
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd -- "$SLURM_SUBMIT_DIR" && pwd)"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  if [[ "${SCRIPT_PATH}" != /* ]]; then
    SCRIPT_PATH="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)/$(basename -- "$SCRIPT_PATH")"
  fi
  SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
fi

cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/slurm/common.sh"

: "${EXPERIMENT_NAME:=qai_train}"
: "${TRAINER:=pytorch}"
: "${HYDRA_OVERRIDES:=}"
export EXPERIMENT_NAME TRAINER REPO_ROOT

log_training_start() {
  python - <<'PY'
from datetime import datetime
import os

from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="QAI Training Start", header_style="bold magenta")
table.add_column("Field", style="bold cyan", no_wrap=True)
table.add_column("Value", style="white")
table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
table.add_row("Job Name", os.environ.get("SLURM_JOB_NAME", "train-qai"))
table.add_row("Job ID", os.environ.get("SLURM_JOB_ID", "local"))
table.add_row("Node List", os.environ.get("SLURM_JOB_NODELIST", "local"))
table.add_row("Experiment", os.environ["EXPERIMENT_NAME"])
table.add_row("Trainer", os.environ["TRAINER"])
table.add_row("Repository", os.environ["REPO_ROOT"])
table.add_row(
    "Command",
    f"python -m src.main experiment={os.environ['EXPERIMENT_NAME']} trainer={os.environ['TRAINER']}",
)
console.print(table)
PY
}

extra_args=()
if [[ -n "$HYDRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  extra_args=($HYDRA_OVERRIDES)
fi

log_training_start

python -m src.main \
  "experiment=${EXPERIMENT_NAME}" \
  "trainer=${TRAINER}" \
  "${extra_args[@]}"
