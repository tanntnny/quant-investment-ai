#!/usr/bin/env bash

#SBATCH --job-name=eval-qai-portfolio-ensemble
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

: "${EXPERIMENT_NAME:=qai_portfolio_multirun_eval}"
: "${HYDRA_OVERRIDES:=}"
: "${EVAL_OUTPUT_DIR:=multirun/2026-04-17/17-44-01}"
: "${MULTIRUN_ROOT:=$EVAL_OUTPUT_DIR}"
: "${RUN_ARTIFACTS_DIR:=saves/run_artifacts/qai_portfolio_best_family_ensemble_eval}"
export EXPERIMENT_NAME REPO_ROOT EVAL_OUTPUT_DIR MULTIRUN_ROOT

log_eval_start() {
  python - <<'PY'
from datetime import datetime
import os

from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="QAI Portfolio Best-Family Ensemble Eval Start", header_style="bold magenta")
table.add_column("Field", style="bold cyan", no_wrap=True)
table.add_column("Value", style="white")
table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
table.add_row("Job Name", os.environ.get("SLURM_JOB_NAME", "eval-qai-portfolio-ensemble"))
table.add_row("Job ID", os.environ.get("SLURM_JOB_ID", "local"))
table.add_row("Node List", os.environ.get("SLURM_JOB_NODELIST", "local"))
table.add_row("Experiment", os.environ["EXPERIMENT_NAME"])
table.add_row("Repository", os.environ["REPO_ROOT"])
table.add_row("Multirun Root", os.environ["MULTIRUN_ROOT"])
table.add_row("Eval Output Dir", os.environ["EVAL_OUTPUT_DIR"])
table.add_row(
    "Command",
    "python -m src.main "
    f"experiment={os.environ['EXPERIMENT_NAME']} "
    f"evaluator.multirun_root={os.environ['MULTIRUN_ROOT']} "
    "evaluator.build_best_family_ensemble=true "
    "evaluator.ensemble_model_families=[attention,bilstm] "
    "evaluator.ensemble_weighting=equal",
)
console.print(table)
PY
}

extra_args=()
if [[ -n "$HYDRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  extra_args=($HYDRA_OVERRIDES)
fi

log_eval_start

cmd=(
  python -m src.main
  "experiment=${EXPERIMENT_NAME}"
  "evaluator.multirun_root=${MULTIRUN_ROOT}"
  "evaluator.build_best_family_ensemble=true"
  "evaluator.ensemble_model_families=[attention,bilstm]"
  "evaluator.ensemble_weighting=equal"
)

if (( ${#extra_args[@]} > 0 )); then
  cmd+=("${extra_args[@]}")
fi

"${cmd[@]}"

mkdir -p "$RUN_ARTIFACTS_DIR"

for artifact_path in \
  metrics_eval.json
do
  if [[ -f "$EVAL_OUTPUT_DIR/$artifact_path" ]]; then
    cp "$EVAL_OUTPUT_DIR/$artifact_path" "$RUN_ARTIFACTS_DIR/$(basename "$artifact_path")"
  fi
done

if [[ -d "$EVAL_OUTPUT_DIR/eval_outputs" ]]; then
  rm -rf "$RUN_ARTIFACTS_DIR/eval_outputs"
  cp -R "$EVAL_OUTPUT_DIR/eval_outputs" "$RUN_ARTIFACTS_DIR/eval_outputs"
fi
