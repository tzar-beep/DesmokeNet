#!/usr/bin/env bash
# Phase 0 go/no-go pilot on SmokeBench: single-image vs reference-guided (v2, v3), 2 seeds each.
# Usage: bash scripts/run_pilot.sh [path/to/SmokeBench.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/.venvs/desmoke/Scripts/python}
DATA=C:/data/smokebench
ZIP=${1:-$DATA/SmokeBench.zip}

if [ ! -f "$DATA/prepared/manifest.csv" ]; then
  [ -d "$DATA/raw" ] || unzip -q "$ZIP" -d "$DATA/raw"
  $PY scripts/prepare_pairs.py --src "$DATA/raw" --out "$DATA/prepared" --max-side 640
fi

M=""
for v in single gated_v2 gated_v3; do
  for s in 0 1; do
    out=runs/pilot_${v}_s$s
    [ -f "$out/best.pt" ] || $PY scripts/train.py --config configs/smokebench_$v.yaml --seed $s --out-dir "$out"
    M="$M $out/best.pt"
  done
done
$PY scripts/evaluate.py --manifest "$DATA/prepared/manifest.csv" --split test --methods identity dcp paste_ref $M --out results/pilot
$PY scripts/gate_diagnostics.py --manifest "$DATA/prepared/manifest.csv" runs/pilot_gated_v*_s*/best.pt
