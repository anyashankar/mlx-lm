#!/bin/bash
# Runs every test on tiny random models (no downloads, CPU is fine), plus the
# CPU build of the CUDA replay against a freshly generated trace.
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/test_spec_decode.py
python tests/test_bench.py
python tests/test_analyze.py
python tests/test_trace_replay.py
echo "ALL TESTS PASSED"
