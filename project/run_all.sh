#!/bin/bash
# Whole pipeline on the M5 Max:   bash run_all.sh          (Granite; or: bash run_all.sh qwen)
#   1. speculative decoding + routing logs + a 6-round trace for the CUDA replay
#   2. verify latency at 1-8 rows, MoE-only time, repeated-expert test
#   3. union statistics, cost table, go/no-go
# Results land in runs/<model>/ (report.md has everything).
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${1:-granite}
case "$MODEL" in
  qwen)
    TARGET=mlx-community/Qwen3-30B-A3B-4bit
    DRAFT=mlx-community/Qwen3-0.6B-4bit
    ;;
  granite)
    # no ready-made 4-bit MLX Granite MoE checkpoints, so convert once
    mkdir -p models
    [ -d models/granite-3b-a800m-4bit ] || mlx_lm.convert --hf-path ibm-granite/granite-3.1-3b-a800m-instruct \
        -q --q-bits 4 --q-group-size 64 --mlx-path models/granite-3b-a800m-4bit
    [ -d models/granite-1b-a400m-4bit ] || mlx_lm.convert --hf-path ibm-granite/granite-3.1-1b-a400m-instruct \
        -q --q-bits 4 --q-group-size 64 --mlx-path models/granite-1b-a400m-4bit
    TARGET=models/granite-3b-a800m-4bit
    DRAFT=models/granite-1b-a400m-4bit
    ;;
  *) echo "usage: $0 qwen|granite"; exit 2 ;;
esac
OUT=runs/$MODEL
mkdir -p "$OUT"

python -c "import mlx.core as mx, mlx_lm, platform; print('mlx', mx.__version__, '| mlx-lm', mlx_lm.__version__, '|', platform.platform(), '| device', mx.default_device())" | tee "$OUT/environment.txt"

python spec_decode.py --target "$TARGET" --draft "$DRAFT" --num-draft 7 --max-tokens 256 \
    --baseline --dump-rounds 6 --out "$OUT" 2>&1 | tee "$OUT/spec_decode.log"
python bench_verify.py --target "$TARGET" --out "$OUT" 2>&1 | tee "$OUT/bench.log"
python analyze.py "$OUT" 2>&1 | tee "$OUT/analyze.log"

echo
echo "Done. Read $OUT/report.md; copy $OUT/trace/ to Bridges-2 for the CUDA replay (cuda/setup_bridges2.sh prints the rsync line)."
