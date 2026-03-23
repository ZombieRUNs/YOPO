#!/usr/bin/env bash
set -euo pipefail

if [[ -d "/root/workspace/YOPO/run/poly-action-rw" ]]; then
  POLY_ROOT="/root/workspace/YOPO/run/poly-action-rw"
elif [[ -d "/root/workspace/poly-action-rw" ]]; then
  POLY_ROOT="/root/workspace/poly-action-rw"
else
  echo "[init] poly-action-rw not found in expected locations." >&2
  echo "[init] tried:" >&2
  echo "       /root/workspace/YOPO/run/poly-action-rw" >&2
  echo "       /root/workspace/poly-action-rw" >&2
  exit 1
fi

ROPE_FILE="${POLY_ROOT}/flowpilot/models/rope3d.py"

if [[ ! -f "${ROPE_FILE}" ]]; then
  echo "[init] rope3d.py not found: ${ROPE_FILE}" >&2
  echo "[init] make sure poly-action-rw is available in container." >&2
  exit 1
fi

if ! grep -q "from __future__ import annotations" "${ROPE_FILE}"; then
  sed -i '1i from __future__ import annotations' "${ROPE_FILE}"
  echo "[init] patched rope3d.py for py3.8 annotations compatibility."
else
  echo "[init] rope3d.py already patched."
fi

export PYTHONPATH="${POLY_ROOT}:/root/workspace/YOPO:${PYTHONPATH:-}"
echo "[init] POLY_ROOT:"
echo "       ${POLY_ROOT}"
echo "[init] PYTHONPATH set to:"
echo "       ${PYTHONPATH}"

echo "[init] verify import..."
python -c "import flowpilot.models.rope3d as r; print('ok', r.__file__)"

echo "[init] done."
