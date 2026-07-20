#!/usr/bin/env bash
# Vendor official YOLOX for .pth weight loading
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/third_party/YOLOX"
if [[ -d "$TARGET/.git" ]]; then
  echo "YOLOX already present at $TARGET"
else
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git "$TARGET"
fi
echo "Done. Install Python deps with: pip install -r requirements.txt"
