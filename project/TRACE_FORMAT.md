# Trace format for the CUDA (and Metal) replay

A trace is a directory written by `spec_decode.py --dump-rounds N`. It holds the
target model's quantized expert weights for some MoE layers, and, for N recorded
verify passes, each layer's expert-block input, chosen experts and outputs.
Both kernels replay the same files, so they see identical routing and weights.

Everything is plain `.npy` (little-endian, C order) plus a `meta.txt` of
`key value` lines, so C++ needs no JSON or npz library (`cuda/npy.h` reads it),
and Python reads it with `np.load`.

## Layout

```
trace/
  meta.txt
  weights/L{l}_{p}_q.npy          uint32  [E, out, in*bits/32]  packed 4- or 8-bit values
  weights/L{l}_{p}_s.npy          float32 [E, out, in/G]     scales
  weights/L{l}_{p}_b.npy          float32 [E, out, in/G]     biases
  weights/L{l}_{p}_e{R}_deq.npy   float32 [out, in]          MLX's dequantization of expert R
  rounds/R{r}_L{l}_x.npy          float32 [rows, hidden]     input to layer l's experts
  rounds/R{r}_L{l}_idx.npy        int32   [rows, top_k]      experts each row picked
  rounds/R{r}_L{l}_y.npy          float32 [rows, top_k, hidden]  per-expert output, as MLX
                                                             computed it in the model dtype
  rounds/R{r}_L{l}_y32.npy        float32 [rows, top_k, hidden]  same output recomputed by
                                                             MLX in float32 (strict reference)
```

`l` is the MoE layer index (0 = first MoE layer), `p` is `gate`, `up` or `down`,
`E` the expert count, `G` the quantization group size, `R` = `ref_expert`.
For gate and up, `out = intermediate`, `in = hidden`; for down it is the reverse.

## meta.txt keys

| key | meaning |
|---|---|
| `num_experts`, `top_k` | E, experts per token |
| `hidden`, `intermediate` | model and expert hidden sizes |
| `bits`, `group_size` | gate projection's bits and group size (kept for old readers) |
| `proj_bits` | bits per projection, e.g. `gate:4 up:4 down:8`. mlx-lm's Granite recipe keeps the expert down projection (and o_proj, router, lm_head) at 8 bits |
| `proj_group` | group size per projection, e.g. `gate:64 up:64 down:64` |
| `activation` | `swiglu`: `y = down( silu(gate x) * (up x) )` |
| `ref_expert` | expert whose dequantized weights are stored for checking |
| `layers` | space-separated MoE layer indices present |
| `rounds` | space-separated `round_id:rows` pairs |
| `dtype` | dtype MLX computed in (`bfloat16` for the 4-bit checkpoints) |
| `decoder_layer_index` | decoder-layer index of each MoE layer |
| `source` | model path the trace came from |

## Quantized weight layout (MLX affine quantization, 4 or 8 bits)

This is MLX's own storage format, unchanged, so nothing is repacked between
MLX and the kernels:

- With `per = 32 / bits` values per word (8 at 4-bit, 4 at 8-bit), row `o` of
  expert `e`, input column `i`, is in word `q[e][o][i/per]`, bits
  `bits*(i%per)` upward: **lowest bits first**.
- `W[e][o][i] = s[e][o][i/G] * value + b[e][o][i/G]`.
- Bit width can differ per projection (`proj_bits`); Granite converted with
  `mlx_lm.convert -q --q-bits 4` has 4-bit gate/up and 8-bit down projections.
- The layer computes `y = W x` (MLX calls `gather_qmm` with `transpose=True`).

Checked against `mx.quantize`/`mx.dequantize` at 4 and 8 bits (exact match), and the CUDA
replay's dequantizer matches the stored `_deq.npy` files exactly.

If the Metal and CUDA kernels settle on a re-tiled layout, convert from this
format when loading; keep the trace in MLX's layout so it stays the single
reference.

## Correctness references

- `_y32` is the pass/fail reference. It uses the same weights (4-bit values
  are exact in any dtype) with float32 activations, scales and biases, and
  MLX's float32 `gather_qmm` matches plain `s*q+b` arithmetic to about 1e-7.
  The CUDA replay reproduces it to about 4e-7 relative error on test traces.
- `_y` is what the model actually produced in bf16. A replay will not match it
  bit for bit: bf16 accumulation differs between MLX backends (MLX's CPU
  backend is about 2% off the exact result in bf16). The replay reports this
  comparison for information only.

## Size

All 48 layers of Qwen3-30B-A3B come to about 18 GB (weights plus float32 scales), so the default
`--dump-layers auto` writes only the first, middle and last MoE layers.
Rounds are small: about 1.2 MB per layer per 8-row round for Qwen3-30B-A3B.
