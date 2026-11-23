"""Checks spec_decode on tiny random models (CPU is fine).

1. Speculative output must equal plain greedy output token for token.
2. Routing is recorded for every target MoE layer on every verify pass, and
   matches the expert ids the router computes for the same rows.
3. A MoE draft model is never recorded.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
import numpy as np

import common
import spec_decode
from tiny_models import perturbed_copy, random_prompts, tiny_granite_moe, tiny_qwen3_moe

mx.set_default_device(mx.cpu)


def check_matches_greedy(target, draft, prompts):
    common.tag_moe_layers(target)
    for ids in prompts:
        for k in (1, 3, 7):
            spec = spec_decode.speculative_greedy(target, draft, ids, k, 40)
            base, _ = spec_decode.greedy_baseline(target, ids, 40)
            assert spec == base, f"k={k}: speculative output differs from greedy\n{spec}\n{base}"
    common.untag(target)
    print("ok: speculative == greedy for k in 1, 3, 7")


def check_routing_and_tags(target, draft, prompts):
    rounds = []
    common.tag_moe_layers(target)
    spec_decode.speculative_greedy(target, draft, prompts[0], 5, 30, on_round=rounds.append)
    L = common.moe_facts(target)["num_moe_layers"]
    for r in rounds:
        layers_seen = sorted(c[0] for c in r["calls"])
        assert layers_seen == list(range(L)), f"recorded layers {layers_seen}, expected 0..{L-1}"
        rows = r["k"] + 1
        for _, idx, _, _ in r["calls"]:
            assert idx.shape[-2] == rows
    # independent check: router top-k for layer 0 on a fresh prefill must match the log
    common.REC.enabled = True
    cache = common.new_cache(target)
    target(mx.array(prompts[0][:8], mx.uint32)[None], cache=cache)
    calls = common.REC.take()
    common.REC.enabled = False
    block = target.model.layers[0].mlp
    h = target.model.embed_tokens(mx.array(prompts[0][:8], mx.uint32)[None])
    h = target.model.layers[0].self_attn(target.model.layers[0].input_layernorm(h), mask="causal", cache=None) + h
    hn = target.model.layers[0].post_attention_layernorm(h)
    gates = mx.softmax(block.gate(hn), axis=-1, precise=True)
    k = block.top_k
    ref = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    got = calls[0][1]
    assert calls[0][0] == 0
    a = np.sort(np.array(ref).reshape(-1, k), 1)
    b = np.sort(np.array(got).reshape(-1, k), 1)
    assert (a == b).all(), "recorded experts differ from the router's own top-k"
    common.untag(target)
    print(f"ok: {len(rounds)} rounds, every MoE layer recorded, layer-0 ids match the router")


def check_run_and_dump(target, draft, prompts):
    with tempfile.TemporaryDirectory() as d:
        args = SimpleNamespace(num_draft=4, max_tokens=24, baseline=True, dump_rounds=2,
                               dump_layers="all", out=d, target="tiny", draft="tiny")
        summary = spec_decode.run(target, draft, None, args, prompts=prompts[:2])
        z = np.load(os.path.join(d, "routing.npz"))
        assert z["row_offsets"][-1] == z["routing"].shape[0]
        assert len(z["k"]) == summary["rounds"]
        assert all(p["matches_greedy"] for p in summary["per_prompt"])
        files = os.listdir(os.path.join(d, "trace", "rounds"))
        L = summary["facts"]["num_moe_layers"]
        # x, idx, y, y32 per layer per round
        assert len(files) == 2 * L * 4, f"expected {2 * L * 4} round files, got {len(files)}"
        meta = open(os.path.join(d, "trace", "meta.txt")).read()
        assert "rounds 0:" in meta, meta
    print("ok: run() writes routing.npz, summary.json and a 2-round trace")


if __name__ == "__main__":
    target = tiny_qwen3_moe()
    draft_dense = perturbed_copy(tiny_qwen3_moe, scale=0.01)  # same arch, noisy
    draft_moe = tiny_granite_moe()  # a MoE draft must not be logged
    prompts = random_prompts(3)

    check_matches_greedy(target, draft_dense, prompts)
    check_matches_greedy(target, draft_moe, prompts[:1])
    check_routing_and_tags(target, draft_moe, prompts)
    check_run_and_dump(target, draft_dense, prompts)
    print("all spec_decode tests passed")
