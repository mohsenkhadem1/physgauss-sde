#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x ../../runs/toolchain/tectonic ]]; then
  ../../runs/toolchain/tectonic -X compile main.tex
  ../../runs/toolchain/tectonic -X compile supplement.tex
elif command -v tectonic >/dev/null; then
  tectonic -X compile main.tex
  tectonic -X compile supplement.tex
else
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
fi
cp supplement.pdf new-study-supplement.pdf
pdfunite new-study-supplement.pdf legacy-supporting-evidence.pdf supplement-combined.pdf
mv supplement-combined.pdf supplement.pdf
