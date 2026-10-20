#!/usr/bin/env bash
set -eu
project_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
isaac_python="${ISAAC_PYTHON:-$project_root/../deformable worldmodel/.venv/bin/python}"
if [ ! -x "$isaac_python" ]; then
    echo 'Set ISAAC_PYTHON to the Python executable of an Isaac Sim 6.2 environment.' >&2
    exit 1
fi
cd "$project_root"
export OMNI_KIT_ACCEPT_EULA=YES
exec "$isaac_python" -u experiments/isaac_boundary_block.py "$@"
