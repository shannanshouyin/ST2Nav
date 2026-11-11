#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
unset CYCLONEDDS_URI
model_args=()
if [[ "${1:-}" == --model ]]; then
    [[ $# -ge 3 && "$3" == navigate ]] || { echo 'Use: ./run.sh --model NAME navigate [options]' >&2; exit 2; }
    model_args=(--model "$2")
    shift 2
fi
if [[ "${1:-}" == navigate ]]; then
    shift
    exec "${ROBOT_PYTHON:-$HOME/.local/share/etpr1/robot-env/bin/python}" -B navigate.py "${model_args[@]}" "$@"
fi
if [[ "${1:-}" == step ]]; then
    shift
    exec "${ROBOT_PYTHON:-$HOME/.local/share/etpr1/robot-env/bin/python}" -B step.py "$@"
fi
if [[ "${1:-}" == dual-scan ]]; then
    shift
    exec "${ROBOT_PYTHON:-$HOME/.local/share/etpr1/robot-env/bin/python}" -B dual_scan.py "$@"
fi
if [[ "${1:-}" == turn || "${1:-}" == scan || "${1:-}" == capture ]]; then
    exec "${ROBOT_PYTHON:-$HOME/.local/share/etpr1/robot-env/bin/python}" -B scan.py "$@"
fi
exec "${ROBOT_PYTHON:-$HOME/.local/share/etpr1/robot-env/bin/python}" -B robot.py "$@"
