#!/usr/bin/env bash
set -euo pipefail
MISHAPE_LAUNCH_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec bash "$MISHAPE_LAUNCH_DIR/start.sh" "$@"
