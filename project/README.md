# Aanya's half of the milestone work

Code for every item on my side of our task list, plus the shared go/no-go.
Everything is tested on tiny random models (`bash tests/run_tests.sh`), but
**nothing here has been run on the real models, the M5 Max, or a real GPU yet**:
those numbers come from running the steps below.

| Task | Where |
|---|---|
| Speculative decoding running (Granite-3.1-3B-A800M + Granite-3.1-1B-A400M; Qwen pair also supported) | `spec_decode.py` |
| Per-layer routing logs | `spec_decode.py` -> `routing.npz` |
| Verify latency at 1-8 rows | `bench_verify.py` -> `bench.json` |
| Expert-union statistics, traffic of repeated expert ids, measured cost table | `bench_verify.py` + `analyze.py` -> `report.md` |
| Trace export format for the CUDA replay | `trace_format.py`, `TRACE_FORMAT.md` (written by `spec_decode.py --dump-rounds`) |
| CUDA environment built (GHC now; Bridges-2 is down until ~December) | `cuda/build_and_test.sh` (records CUDA version and GPU), `cuda/setup_bridges2.sh` for later |
| CUDA litmus tests | `cuda/litmus.cu` |
| CUDA trace-replay harness, quantized weights with dequantization (4- and 8-bit; Granite's expert down projection is 8-bit) | `cuda/replay.cu`, `cuda/npy.h` |
| Go/no-go from the union statistics | `analyze.py` (verdict at the top of `report.md`) |

## 1. On the Mac (M5 Max)

One-time setup (in the unzipped `aanya/` folder):

```bash
python3 -m venv .venv && source .venv/bin/activate   # or reuse the env that has mlx 0.32.4
pip install "mlx==0.32.4" mlx-lm matplotlib
python -c "import mlx.core as mx; print(mx.__version__, mx.default_device())"   # 0.32.4 Device(gpu, 0)
xcode-select --install          # only if g++ is missing; the replay test compiles C++
bash tests/run_tests.sh         # a few minutes on tiny models; must end with ALL TESTS PASSED
```

Then run everything (Granite by default):

```bash
bash run_all.sh                 # or: bash run_all.sh qwen
```

The first Granite run downloads `ibm-granite/granite-3.1-3b-a800m-instruct` and
`granite-3.1-1b-a400m-instruct` from Hugging Face (several GB in bf16) and
converts both to 4-bit MLX in `models/` (`mlx_lm.convert -q --q-bits 4
--q-group-size 64`). Later runs reuse the converted models. Leave around 15 GB free.

`run_all.sh` runs three steps and writes everything to `runs/<model>/`:
1. `spec_decode.py`: greedy speculative decoding over 16 varied prompts, 7 drafts
   per round. Logs, for every verify pass, the experts each row picked in every
   MoE layer, the draft model's probability for each draft, how many were
   accepted, and draft/verify time. `--baseline` also runs plain greedy
   decoding, so you get the speedup and a check that the output is identical.
   `--dump-rounds 6` writes the CUDA replay trace (first, middle and last MoE
   layers by default).
2. `bench_verify.py`: verify latency at 1-8 rows on real contexts (median of
   15, shuffled order), MoE-only time (just the expert matmuls), and a
   repeated-expert test (same vs. distinct experts at equal row-expert pairs,
   which shows whether MLX re-reads repeated experts).
3. `analyze.py`: union statistics, acceptance and calibration, the cost table,
   a fitted cost model, replays of every logged round under each cut policy,
   the go/no-go, and three figures for the report.

Read `runs/<model>/report.md` first. It is written for the milestone report.

## 2. How the go/no-go is decided

`analyze.py` replays the logged rounds under five cut policies and compares
realized tokens per millisecond, using the measured cost model:

- **fixed-all / fixed-best**: always verify all rows / the best fixed count.
- **pre-pass** (EVICT-style): per round, maximize expected tokens / *average*
  cost of r rows.
- **in-pass@l**: decide after seeing the first l MoE layers' routing; earlier
  layers already ran with all rows; later layers are extrapolated.
- **in-pass-oracle**: the same, with the round's full routing known (upper bound).

**GO** if the best realistic in-pass policy beats pre-pass by at least 5%
(`--threshold`). **MARGINAL** if only the oracle does: the per-round
information exists, but early layers don't predict later ones. **NO-GO** if
even the oracle doesn't. Then the proposal's fallback applies. The 5% is our
choice; say so in the report and change it if Anya disagrees.

Expected tokens use draft probabilities **calibrated** to observed acceptance
(fit on half the prompts, scored on the other half). On the test models, raw
draft probabilities were so far off that every policy fell apart. `--value draft`
uses raw probabilities instead, which is what EVICT does. Both are worth
reporting.

Simplifications to state in the report: one cut point per round (the real
kernel can cut at every layer), and attention for dropped rows is charged in
proportion to layers.

## 3. CUDA: GHC machines now, Bridges-2 later

Bridges-2 is down for its data-center move (from Nov 9 for about 4-6 weeks,
back around December), so it may not return before the Dec 10 final report.
The CUDA code runs on any NVIDIA GPU, including the GHC machines we used for
A3/A4. `build_and_test.sh` works on either and records which GPU it ran on.

```bash
# on the Mac: copy the code and the trace (off campus: VPN, or hop via unix.andrew.cmu.edu)
scp -r cuda <andrewid>@ghc73.ghc.andrew.cmu.edu:~/moe/
rsync -av runs/granite/trace/ <andrewid>@ghc73.ghc.andrew.cmu.edu:~/moe/cuda/trace/

# on the GHC machine
nvidia-smi                      # confirm a GPU is visible and see what else is running on it
cd ~/moe/cuda && bash build_and_test.sh trace
```

If AFS quota is tight (it was during A4), put the trace in `/tmp` on that
machine instead: `rsync ... ghc73...:/tmp/<andrewid>-trace/`, then
`bash build_and_test.sh /tmp/<andrewid>-trace`. `/tmp` is local and not kept.

GHC machines are shared: another user's GPU job changes litmus timing, so
note `nvidia-smi` output with the results (the script saves it). Replay
correctness does not depend on load.

When Bridges-2 returns (check psc.edu for the date):
`bash setup_bridges2.sh` on the login node shows your charge ID, then
`interact -p GPU-shared --gres=gpu:v100-32:1 -t 01:00:00 -A <chargeID>` (or
`h100-80`) and `bash build_and_test.sh trace`, or `sbatch -A <chargeID> run_bridges2.sbatch trace`.

**Litmus tests** (`litmus.cu`): MP, SB and LB, with the two threads in
different blocks (the analogue of Metal threadgroups). Each runs with relaxed
atomics, plain loads after an L1 warm-up (the non-coherent-L1 case), a
block-scope fence, a device-scope fence, and acquire/release or seq_cst
ordering. A weak outcome where the PTX model forbids one is flagged as a
violation. Seeing zero weak outcomes for the allowed cases is a legitimate
result, not a bug.

**Trace replay** (`replay.cu`): checks the dequantizer (4- and 8-bit) alone against
MLX's dequantization, then recomputes every recorded expert output. The pass/fail
check compares against MLX's float32 recompute (`_y32`, tolerance 1e-4). The
comparison with MLX's actual bf16 output is informational: bf16 accumulation
differs between MLX backends, so expect about 1-3% max error there and cosine
above 0.999.

## 4. Tests

```bash
bash tests/run_tests.sh
```

Tiny random Qwen3-MoE/Granite-MoE models, CPU, no downloads. They check:

- speculative output equals greedy output at 1, 3 and 7 drafts;
- every MoE layer is logged on every verify pass, and the logged ids match the router's own top-k;
- a MoE draft model is never logged;
- union and expected-token math match brute force;
- the full run -> bench -> analyze pipeline works;
- the CPU build of `replay.cu` passes on a clean bf16 trace and fails on a corrupted one.

`replay.cu` and `litmus.cu` were compiled for sm_70 and sm_90 down to GPU
machine code (clang + ptxas), but never run on a GPU.

## For Anya

- Granite's experts are mixed precision after `mlx_lm.convert`: gate/up 4-bit, down 8-bit. The fused kernel must handle both (or the model must be re-quantized all-4-bit on purpose).
- The trace (`TRACE_FORMAT.md`) is in MLX's own layout, so the Metal
  kernel can replay exactly the same rounds; `_y32` is the shared correctness reference.
- If you fix a re-tiled layout for both kernels, keep the trace in MLX layout and
  convert at load time.
- `common.py`'s `REC` recorder captures (layer, expert ids, input, output) for
  any mlx-lm MoE model, if the stock-MLX truncation needs routing too.
