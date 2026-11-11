#!/bin/sh
# 使い方: ./build.sh src/ch2_practice.py
# ソースの【...】は穴埋め。dist/ には ___ にした配布用、check/ には答えを入れて実行したものを作る。
set -e
name=$(basename "$1" .py)
mkdir -p dist check
uv run jupytext --to ipynb -o "check/$name.ipynb" "$1"
uv run python - "$name" <<'EOF'
import re, sys
from pathlib import Path
name = sys.argv[1]
src = Path(f"check/{name}.ipynb").read_text()
Path(f"dist/{name}.ipynb").write_text(re.sub("【.*?】", "___", src))
Path(f"check/{name}.ipynb").write_text(re.sub("【(.*?)】", r"\1", src))
EOF
(cd check && uv run jupyter nbconvert --to notebook --execute --inplace "$name.ipynb" --ExecutePreprocessor.timeout=1200)
