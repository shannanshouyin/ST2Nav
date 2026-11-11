#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${ROBOT_PYTHON:-$HOME/.local/share/etpr1/robot-env/bin/python}" -B infer.py "$@"
