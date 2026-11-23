"""Writer for the CUDA replay trace. The format is documented in TRACE_FORMAT.md.

Everything is plain .npy (little-endian, C order) plus a meta.txt of
`key value` lines, so the C++ side needs no JSON or npz library.
"""

import os

import numpy as np


def _np(a, dtype=None):
    import mlx.core as mx

    if dtype is not None:
        a = a.astype(dtype)
    return np.array(a)


def fp32_expert_output(mod, x, idx):
    """SwitchGLU's math with float32 activations, scales and biases.

    Same weights (4-bit values are exact in any dtype), so this is the
    precision-independent answer the CUDA replay must reproduce; MLX's
    gather_qmm in float32 matches s*q+b arithmetic to ~1e-7.
    """
    import mlx.core as mx
    import mlx.nn as nn

    f32 = mx.float32
    h = mx.expand_dims(x.astype(f32), (-2, -3))

    def proj(p, v):
        return mx.gather_qmm(v, p.weight, p.scales.astype(f32), p.biases.astype(f32),
                             rhs_indices=idx, transpose=True, group_size=p.group_size, bits=p.bits)

    act = nn.silu(proj(mod.gate_proj, h)) * proj(mod.up_proj, h)
    return proj(mod.down_proj, act).squeeze(-2)


class TraceWriter:
    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "weights"), exist_ok=True)
        os.makedirs(os.path.join(root, "rounds"), exist_ok=True)
        self.meta = {}
        self.rounds = []  # (round_id, rows)

    # -- model and weights ---------------------------------------------------
    def write_model(self, model, facts, top_k, layers=None, ref_expert=0):
        """Write the 4-bit expert weights of the chosen MoE layers.

        Per layer l and projection p in {gate, up, down}:
          weights/L{l}_{p}_q.npy   uint32  [E, out, in*bits/32]  packed values (4- or 8-bit;
                                                                  per projection, see proj_bits)
          weights/L{l}_{p}_s.npy   float32 [E, out, in/group]    scales
          weights/L{l}_{p}_b.npy   float32 [E, out, in/group]    biases
        plus weights/L{l}_{p}_e{ref}_deq.npy, MLX's own dequantization of one
        expert, so the CUDA dequantizer can be checked on its own.
        """
        import mlx.core as mx

        from common import find_moe_layers

        if not facts["quantized"]:
            raise RuntimeError("trace export expects a quantized model (mlx_lm.convert -q)")
        if facts["quant_mode"] not in (None, "affine"):
            raise RuntimeError(f"quantization mode {facts['quant_mode']!r} not supported; use affine")

        all_layers = find_moe_layers(model)
        chosen = list(range(len(all_layers))) if layers is None else list(layers)
        for l in chosen:
            _, _, mod = all_layers[l]
            for pname in ("gate", "up", "down"):
                p = getattr(mod, f"{pname}_proj")
                base = os.path.join(self.root, "weights", f"L{l}_{pname}")
                np.save(base + "_q.npy", _np(p.weight))
                np.save(base + "_s.npy", _np(p.scales, mx.float32))
                if p.biases is None:
                    raise RuntimeError("affine quantization without biases is unexpected")
                np.save(base + "_b.npy", _np(p.biases, mx.float32))
                # float32 scales/biases so the reference is s*q+b exactly; with
                # bf16 scales MLX would round the result to bf16
                deq = mx.dequantize(
                    p.weight[ref_expert], p.scales[ref_expert].astype(mx.float32),
                    p.biases[ref_expert].astype(mx.float32),
                    group_size=p.group_size, bits=p.bits,
                )
                np.save(base + f"_e{ref_expert}_deq.npy", _np(deq, mx.float32))

        pb, pg = facts["proj_bits"], facts["proj_group"]
        for n in ("gate", "up", "down"):
            if pb[n] not in (4, 8):
                raise RuntimeError(f"{n}_proj is {pb[n]}-bit; the replay supports 4 and 8")
        self.meta.update(
            num_experts=facts["num_experts"],
            top_k=top_k,
            hidden=facts["hidden"],
            intermediate=facts["intermediate"],
            bits=facts["bits"],
            group_size=facts["group_size"],
            proj_bits=" ".join(f"{n}:{pb[n]}" for n in ("gate", "up", "down")),
            proj_group=" ".join(f"{n}:{pg[n]}" for n in ("gate", "up", "down")),
            activation="swiglu",  # y = down( silu(gate x) * (up x) )
            ref_expert=ref_expert,
            layers=" ".join(str(l) for l in chosen),
        )

    # -- recorded verify rounds ----------------------------------------------
    def write_round(self, round_id, calls, layers=None, modules=None):
        """calls: RoutingRecorder output for one verify pass, with tensors.

        rounds/R{r}_L{l}_x.npy    float32 [rows, hidden]        expert-block input
        rounds/R{r}_L{l}_idx.npy  int32   [rows, top_k]         chosen experts
        rounds/R{r}_L{l}_y.npy    float32 [rows, top_k, hidden] per-expert output, as MLX
                                                                 computed it (model dtype)
        rounds/R{r}_L{l}_y32.npy  float32 [rows, top_k, hidden] the same output recomputed
                                                                 by MLX entirely in float32:
                                                                 the strict replay reference
        modules: {moe_layer: SwitchGLU}, needed for y32.
        """
        import mlx.core as mx

        rows = None
        for layer, idx, x, y in calls:
            if layers is not None and layer not in layers:
                continue
            if x is None:
                raise RuntimeError("round dump needs REC.capture_tensors = True")
            base = os.path.join(self.root, "rounds", f"R{round_id}_L{layer}")
            xa = _np(x.reshape(-1, x.shape[-1]), mx.float32)
            ia = _np(idx.reshape(-1, idx.shape[-1]), mx.int32)
            ya = _np(y.reshape(-1, y.shape[-2], y.shape[-1]), mx.float32)
            rows = xa.shape[0]
            # dtype MLX computed in; the replay emulates its bf16 rounding
            self.meta["dtype"] = str(x.dtype).replace("mlx.core.", "")
            np.save(base + "_x.npy", xa)
            np.save(base + "_idx.npy", ia)
            np.save(base + "_y.npy", ya)
            if modules is not None:
                y32 = fp32_expert_output(modules[layer], x, idx)
                np.save(base + "_y32.npy", _np(y32.reshape(ya.shape)))
        if rows is not None:
            self.rounds.append((round_id, rows))

    def close(self, extra=None):
        if extra:
            self.meta.update(extra)
        self.meta["rounds"] = " ".join(f"{r}:{n}" for r, n in self.rounds)
        with open(os.path.join(self.root, "meta.txt"), "w") as f:
            for k, v in self.meta.items():
                f.write(f"{k} {v}\n")
