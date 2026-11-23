"""Shared helpers: model loading, routing capture, cache helpers, expert sizes.

Routing capture works by wrapping mlx_lm's SwitchGLU.__call__, which every
MoE model in mlx-lm (qwen3_moe, granitemoe, mixtral, olmoe, ...) uses for its
expert matmuls. Only SwitchGLU instances that tag_moe_layers() marked on the
*target* model are recorded, so a MoE draft model is never logged by mistake.
"""

import re
import time

import mlx.core as mx
import mlx_lm.models.switch_layers as SL
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


# --------------------------------------------------------------------------
# Routing recorder
# --------------------------------------------------------------------------
class RoutingRecorder:
    """Collects (moe_layer, indices, x, y) for each tagged SwitchGLU call.

    enabled          -- record anything at all
    capture_tensors  -- also keep the expert input x and output y (for the
                        CUDA replay trace); off by default because y is large
    Arrays are kept lazy; call take() to evaluate and clear them.
    """

    def __init__(self):
        self.enabled = False
        self.capture_tensors = False
        self.calls = []

    def take(self):
        calls, self.calls = self.calls, []
        flat = []
        for _, idx, x, y in calls:
            flat.append(idx)
            if x is not None:
                flat += [x, y]
        if flat:
            mx.eval(flat)
        return calls


REC = RoutingRecorder()
_ORIG_SWITCHGLU_CALL = SL.SwitchGLU.__call__


def _recording_call(self, x, indices):
    y = _ORIG_SWITCHGLU_CALL(self, x, indices)
    layer = getattr(self, "_trace_layer", None)
    if REC.enabled and layer is not None:
        if REC.capture_tensors:
            REC.calls.append((layer, indices, x, y))
        else:
            REC.calls.append((layer, indices, None, None))
    return y


SL.SwitchGLU.__call__ = _recording_call


def find_moe_layers(model):
    """Return [(moe_index, decoder_layer_index, SwitchGLU module)] in layer order."""
    found = []
    for name, mod in model.named_modules():
        if isinstance(mod, SL.SwitchGLU):
            m = re.search(r"layers\.(\d+)\.", name + ".")
            if m is None:
                raise RuntimeError(f"cannot find a layer index in module name {name!r}")
            found.append((int(m.group(1)), mod))
    found.sort(key=lambda t: t[0])
    return [(i, layer_idx, mod) for i, (layer_idx, mod) in enumerate(found)]


def tag_moe_layers(model):
    """Mark the target model's SwitchGLU modules so the recorder logs them."""
    layers = find_moe_layers(model)
    if not layers:
        raise RuntimeError("target model has no SwitchGLU (MoE) layers")
    for moe_idx, _, mod in layers:
        mod._trace_layer = moe_idx
    return layers


def untag(model):
    for _, _, mod in find_moe_layers(model):
        if hasattr(mod, "_trace_layer"):
            del mod._trace_layer


# --------------------------------------------------------------------------
# Model facts
# --------------------------------------------------------------------------
def moe_facts(model):
    """Expert count, top-k, sizes, and bytes per expert for each MoE layer."""
    layers = find_moe_layers(model)
    _, _, first = layers[0]
    gp = first.gate_proj
    quantized = isinstance(gp, SL.QuantizedSwitchLinear)
    num_experts = gp.weight.shape[0]
    hidden = gp.input_dims if quantized else gp.weight.shape[2]
    inter = gp.output_dims if quantized else gp.weight.shape[1]

    def proj_bytes(p):
        n = p.weight.nbytes
        if quantized:
            n += p.scales.nbytes
            if getattr(p, "biases", None) is not None:
                n += p.biases.nbytes
        if "bias" in p:
            n += p["bias"].nbytes
        return n

    per_layer = []
    for _, layer_idx, mod in layers:
        total = sum(proj_bytes(p) for p in (mod.gate_proj, mod.up_proj, mod.down_proj))
        per_layer.append(total / num_experts)

    return {
        "num_moe_layers": len(layers),
        "decoder_layer_index": [li for _, li, _ in layers],
        "num_experts": int(num_experts),
        "hidden": int(hidden),
        "intermediate": int(inter),
        "quantized": quantized,
        "bits": int(gp.bits) if quantized else None,  # gate projection; see proj_bits
        "group_size": int(gp.group_size) if quantized else None,
        # mlx-lm's Granite recipe keeps down_proj at 8 bits, so record each projection
        "proj_bits": {n: int(getattr(first, f"{n}_proj").bits) for n in ("gate", "up", "down")} if quantized else None,
        "proj_group": {n: int(getattr(first, f"{n}_proj").group_size) for n in ("gate", "up", "down")} if quantized else None,
        "quant_mode": getattr(gp, "mode", None) if quantized else None,
        "bytes_per_expert": per_layer,  # one value per MoE layer
    }


def top_k_of(model):
    """Experts per token, read from the model config."""
    args = getattr(model, "args", None)
    for key in ("num_experts_per_tok", "top_k", "moe_top_k"):
        v = getattr(args, key, None)
        if v:
            return int(v)
    raise RuntimeError("cannot find experts-per-token in model.args")


# --------------------------------------------------------------------------
# Loading and prompts
# --------------------------------------------------------------------------
def load_pair(target_path, draft_path):
    from mlx_lm import load

    target, tok = load(target_path)
    draft, dtok = load(draft_path)
    if tok.vocab_size != dtok.vocab_size:
        print(
            f"WARNING: tokenizer vocab sizes differ (target {tok.vocab_size}, "
            f"draft {dtok.vocab_size}); speculative decoding needs a shared tokenizer."
        )
    return target, draft, tok


def encode_prompt(tok, text, chat=True):
    if chat and getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": text}]
        try:
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        if isinstance(ids, str):
            ids = tok.encode(ids, add_special_tokens=False)
        return list(ids)
    return tok.encode(text)


# Varied on purpose: routing depends on content, so a single genre would
# understate how much expert usage changes between rounds.
PROMPTS = [
    "Write a Python function that merges two sorted linked lists, with comments.",
    "Explain how a CPU cache works to a first-year computer science student.",
    "Solve step by step: a train leaves at 3:15 pm going 80 km/h; a second leaves "
    "the same station at 4:00 pm going 110 km/h. When does the second catch up?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Summarize the causes and main consequences of the French Revolution.",
    "Translate into French and then Spanish: 'The meeting has been moved to Thursday "
    "afternoon because the projector is broken.'",
    "Write a SQL query that returns, for each department, the employee with the "
    "highest salary, handling ties.",
    "What are the trade-offs between TCP and UDP for a multiplayer game?",
    "Write a haiku sequence (five haiku) about winter in Pittsburgh.",
    "Give a recipe for vegetable biryani with quantities for four people.",
    "Implement a thread-safe bounded queue in C++ using a mutex and two condition variables.",
    "Compare the economic policies of Keynesian and monetarist schools in a table.",
    "Explain what a Mixture-of-Experts language model is and why it is cheaper to run.",
    "Write a cover letter opening paragraph for a software engineering internship.",
    "Prove that the square root of 2 is irrational.",
    "List ten everyday uses of the Fourier transform and explain each in one sentence.",
]


# --------------------------------------------------------------------------
# Cache helpers
# --------------------------------------------------------------------------
def prefill(model, cache, ids, step=512):
    """Run ids[:-1] into cache; return the last token as an array of shape (1,)."""
    y = mx.array(ids, mx.uint32)
    while y.size > 1:
        n = min(step, y.size - 1)
        model(y[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[n:]
    return y


def new_cache(model):
    return make_prompt_cache(model)


def trim(cache, n):
    if n > 0:
        trim_prompt_cache(cache, n)


def sync_time(fn):
    """Time fn() including GPU completion. fn must return arrays to evaluate."""
    mx.synchronize()
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    return time.perf_counter() - t0, out
