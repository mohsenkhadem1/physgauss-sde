#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
repo=runs/third_party/PhysTwin
revision=81c718790a37e5e0102eb77af2c6edd34a9db25f
if [ ! -d "$repo/.git" ]; then
    git clone https://github.com/Jianghanxiao/PhysTwin.git "$repo"
    git -C "$repo" checkout "$revision"
fi
[ "$(git -C "$repo" rev-parse HEAD)" = "$revision" ] || { echo 'Unexpected PhysTwin revision'; exit 1; }
if git -C "$repo" apply --check ../../../scripts/phystwin/headless-imports.patch 2>/dev/null; then
    git -C "$repo" apply ../../../scripts/phystwin/headless-imports.patch
else
    git -C "$repo" apply --reverse --check ../../../scripts/phystwin/headless-imports.patch
fi
if [ ! -x runs/phystwin-env/bin/python ]; then
    uv venv --python 3.10 runs/phystwin-env
fi
uv pip install --python runs/phystwin-env/bin/python torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python runs/phystwin-env/bin/python -r scripts/phystwin/requirements.txt
runs/phystwin-env/bin/python scripts/phystwin/fetch_example.py --case "${1:-single_lift_sloth}"
