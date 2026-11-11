#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
exec "${SERVER_PYTHON:-python}" -B server.py --model st2nav "$@"
