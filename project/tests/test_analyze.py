"""End to end on tiny models: spec_decode -> bench_verify -> analyze.

Also checks the union and expected-token math against brute force.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
import numpy as np

import analyze
import bench_verify
import spec_decode
from tiny_models import perturbed_copy, random_prompts, tiny_qwen3_moe

mx.set_default_device(mx.cpu)


def check_math():
    rng = np.random.default_rng(0)
    rows, L, k, E = 6, 3, 4, 16
    R = np.stack([np.stack([rng.choice(E, k, replace=False) for _ in range(L)]) for _ in range(rows)])
    U = analyze.prefix_unions(R.astype(np.int16), np.array([0, rows]), E)[0]
    for r in range(1, rows + 1):
        for l in range(L):
            assert U[r - 1, l] == len(set(R[:r, l].ravel())), (r, l)
    p = np.array([0.9, 0.5, 0.2])
    assert analyze.expected_tokens(p, 1) == 1.0
    assert abs(analyze.expected_tokens(p, 4) - (1 + 0.9 + 0.45 + 0.09)) < 1e-9
    print("ok: prefix unions and expected tokens match brute force")


if __name__ == "__main__":
    check_math()
    target = tiny_qwen3_moe(layers=8)
    draft = perturbed_copy(tiny_qwen3_moe, scale=0.01, layers=8)
    prompts = random_prompts(4)
    with tempfile.TemporaryDirectory() as d:
        spec_decode.run(target, draft, None, SimpleNamespace(
            num_draft=5, max_tokens=40, baseline=False, dump_rounds=0, dump_layers="all",
            out=d, target="tiny", draft="tiny"), prompts=prompts)
        bench_verify.run(target, prompts[:2], SimpleNamespace(max_rows=6, reps=3, micro_layer=0, out=d))
        res = analyze.analyze(d)
        assert res["verdict"][0] in ("GO", "MARGINAL", "NO-GO")
        assert set(res["policies"]) >= {"fixed-all", "fixed-best", "pre-pass", "in-pass-oracle"}
        # the oracle has strictly more information than pre-pass on the same value term
        assert res["policies"]["in-pass-oracle"]["tok_per_ms"] >= res["policies"]["pre-pass"]["tok_per_ms"] * 0.999
        figs = [f for f in os.listdir(d) if f.startswith("fig_")]
        assert len(figs) == 3, figs
    print("analyze end-to-end test passed")
