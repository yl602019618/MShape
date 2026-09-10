#!/usr/bin/env bash
# MiShape local launcher. Does not start or modify the legacy AeroShape app.
set -euo pipefail
MISHAPE_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -f "$MISHAPE_SCRIPT_DIR/app.py" && -d "$MISHAPE_SCRIPT_DIR/../aeroshape" ]]; then
  MISHAPE_PROJECT_DIR="$(CDPATH= cd -- "$MISHAPE_SCRIPT_DIR/.." && pwd)"
elif [[ -f "$MISHAPE_SCRIPT_DIR/mishape/app.py" && -d "$MISHAPE_SCRIPT_DIR/aeroshape" ]]; then
  MISHAPE_PROJECT_DIR="$MISHAPE_SCRIPT_DIR"
else
  echo "无法找到 MiShape 和所需的 aeroshape 几何模块。请保留交付目录结构。" >&2
  exit 1
fi
MISHAPE_REQUIREMENTS="$MISHAPE_PROJECT_DIR/mishape/requirements.txt"
MISHAPE_URL="http://127.0.0.1:8010"
MISHAPE_CHECK=0
if [[ "${1:-}" == "--check" ]]; then MISHAPE_CHECK=1; fi
if [[ $# -gt 1 || ( $# -eq 1 && "$MISHAPE_CHECK" == 0 ) ]]; then
  echo "用法：bash start.sh [--check]" >&2
  exit 2
fi
cd "$MISHAPE_PROJECT_DIR"
MISHAPE_PYTHON=""
for MISHAPE_CANDIDATE in "$MISHAPE_PROJECT_DIR/.venv/bin/python" "$MISHAPE_SCRIPT_DIR/.venv/bin/python"; do
  if [[ -x "$MISHAPE_CANDIDATE" ]] && "$MISHAPE_CANDIDATE" -c 'import sys; raise SystemExit(sys.version_info < (3,11))' >/dev/null 2>&1; then
    MISHAPE_PYTHON="$MISHAPE_CANDIDATE"
    break
  fi
done
if [[ -z "$MISHAPE_PYTHON" ]]; then
  if [[ "$MISHAPE_CHECK" == 1 ]]; then
    echo "未找到 Python 3.11+ 虚拟环境。运行 bash start.sh 创建环境并安装依赖。" >&2
    exit 1
  fi
  MISHAPE_BASE_PYTHON=""
  for MISHAPE_CANDIDATE in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$MISHAPE_CANDIDATE" >/dev/null 2>&1 && "$MISHAPE_CANDIDATE" -c 'import sys; raise SystemExit(sys.version_info < (3,11))' >/dev/null 2>&1; then
      MISHAPE_BASE_PYTHON="$(command -v "$MISHAPE_CANDIDATE")"
      break
    fi
  done
  if [[ -z "$MISHAPE_BASE_PYTHON" ]]; then
    echo "MiShape 需要 Python 3.11 或更新版本。安装后重新运行此启动器。" >&2
    exit 1
  fi
  echo "正在创建 MiShape 虚拟环境……"
  "$MISHAPE_BASE_PYTHON" -m venv "$MISHAPE_SCRIPT_DIR/.venv"
  MISHAPE_PYTHON="$MISHAPE_SCRIPT_DIR/.venv/bin/python"
fi
if ! "$MISHAPE_PYTHON" -c 'import numpy, numba, scipy, trimesh, networkx, fastapi, pydantic, uvicorn, multipart, PIL; from mishape.generation import generation_schema; generation_schema()' >/dev/null 2>&1; then
  if [[ "$MISHAPE_CHECK" == 1 ]]; then
    echo "运行依赖尚未就绪；运行 bash start.sh 将安装 requirements.txt。" >&2
    exit 1
  fi
  echo "正在安装 MiShape 依赖；首次运行需要联网……"
  if ! "$MISHAPE_PYTHON" -m pip --version >/dev/null 2>&1; then
    "$MISHAPE_PYTHON" -m ensurepip --upgrade
  fi
  "$MISHAPE_PYTHON" -m pip install -r "$MISHAPE_REQUIREMENTS"
fi
if [[ "$MISHAPE_CHECK" == 1 ]]; then
  echo "MiShape 启动检查通过。"
  echo "Python: $MISHAPE_PYTHON"
  echo "目录: $MISHAPE_PROJECT_DIR"
  echo "地址: $MISHAPE_URL"
  exit 0
fi
# Reuse an already running MiShape instance. Do not stop another local service.
if "$MISHAPE_PYTHON" - <<'PY' >/dev/null 2>&1
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:8010/api/health',timeout=.5) as response:
    data=json.load(response)
assert data.get('product')=='MiShape' and data.get('status')=='ok'
PY
then
  echo "MiShape 已在 $MISHAPE_URL 运行。"
  if command -v open >/dev/null 2>&1; then open "$MISHAPE_URL"; fi
  exit 0
fi
export MISHAPE_STATE="${MISHAPE_STATE:-$MISHAPE_PROJECT_DIR/mishape/.state}"
echo "MiShape — polygon 车辆设计工作台"
echo "打开 $MISHAPE_URL；在此终端按 Ctrl+C 停止服务。"
# Wait for health before opening the local page; this helper never kills services.
if command -v open >/dev/null 2>&1; then
  "$MISHAPE_PYTHON" - <<'PY' >/dev/null 2>&1 &
import json, subprocess, time, urllib.request
for _ in range(40):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8010/api/health',timeout=.5) as response:
            data=json.load(response)
        if data.get('product')=='MiShape':
            subprocess.run(['open','http://127.0.0.1:8010'],check=False)
            break
    except Exception:
        time.sleep(.5)
PY
fi
exec "$MISHAPE_PYTHON" -m uvicorn mishape.app:app --host 127.0.0.1 --port 8010
