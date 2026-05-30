#!/usr/bin/env bash
# Bring AKOrN in as a PINNED dependency (cloned to external/akorn, gitignored) and install deps.
# We do NOT vendor AKOrN into this repo's git history (license + drift). Our code imports from it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AKORN_DIR="$ROOT/external/akorn"
AKORN_COMMIT="eabbe2728c5b83836ce44f6d780b7cc37eea2a31"   # pinned: what the scaffolding was read against

mkdir -p "$ROOT/external"
if [ ! -d "$AKORN_DIR/.git" ]; then
  git clone https://github.com/autonomousvision/akorn.git "$AKORN_DIR"
fi
git -C "$AKORN_DIR" fetch origin "$AKORN_COMMIT" --depth 1 2>/dev/null || git -C "$AKORN_DIR" fetch origin
git -C "$AKORN_DIR" checkout -q "$AKORN_COMMIT"
echo "AKOrN pinned at $AKORN_COMMIT -> $AKORN_DIR"

# Lean deps for the classification + CL study (NOT AKOrN's full requirements.txt, which pulls
# tensorflow/auto-attack/pycocotools we don't need). Install torch matching your CUDA first if needed.
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"
echo "Setup done. Then:  source env.sh   (or rely on experiments/m1_wk0/_bootstrap.py)"
