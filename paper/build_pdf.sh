#!/usr/bin/env bash
# Build manuscript.pdf from manuscript.md.
#   markdown --(pandoc, MathML)--> HTML --(headless Chromium)--> PDF
# No LaTeX toolchain required.
set -euo pipefail

cd "$(dirname "$0")"

CHROME="${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}"
SRC="${1:-manuscript.md}"
BASE="${SRC%.md}"

echo "[1/2] pandoc: $SRC -> $BASE.html"
pandoc "$SRC" \
  --from=markdown+tex_math_dollars+pipe_tables+backtick_code_blocks \
  --to=html5 \
  --mathml \
  --standalone \
  --css=style.css \
  --metadata title="" \
  --output="$BASE.html"

echo "[2/2] chromium: $BASE.html -> $BASE.pdf"
"$CHROME" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --no-pdf-header-footer \
  --virtual-time-budget=15000 \
  --print-to-pdf="$BASE.pdf" \
  "file://$PWD/$BASE.html" 2>/dev/null

if [[ -f "$BASE.pdf" ]]; then
  echo "OK -> $PWD/$BASE.pdf ($(du -h "$BASE.pdf" | cut -f1))"
else
  echo "FAILED: no PDF produced" >&2
  exit 1
fi
