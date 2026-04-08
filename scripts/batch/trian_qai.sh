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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/scripts/slurm/common.sh"

: "${EXPERIMENT_NAME:=qai_train}"
: "${TRAINER:=pytorch}"
: "${HYDRA_OVERRIDES:=}"

extra_args=()
if [[ -n "$HYDRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  extra_args=($HYDRA_OVERRIDES)
fi

python -m src.main \
  "experiment=${EXPERIMENT_NAME}" \
  "trainer=${TRAINER}" \
  "${extra_args[@]}"
