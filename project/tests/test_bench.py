import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import json

import mlx.core as mx

import bench_verify
from tiny_models import random_prompts, tiny_qwen3_moe

mx.set_default_device(mx.cpu)

if __name__ == "__main__":
    target = tiny_qwen3_moe()
    with tempfile.TemporaryDirectory() as d:
        args = SimpleNamespace(max_rows=8, reps=3, micro_layer=1, out=d)
        bench_verify.run(target, random_prompts(2), args)
        b = json.load(open(os.path.join(d, "bench.json")))
        assert len(b["verify"]) == 2 * 8
        for rec in b["verify"]:
            assert len(rec["unique_experts_per_layer"]) == 4
            assert 1 <= max(rec["unique_experts_per_layer"]) <= min(16, rec["rows"] * 4)
        r1 = [x for x in b["verify"] if x["rows"] == 1][0]
        assert r1["unique_experts_per_layer"] == [4, 4, 4, 4], "1 row must read exactly top-k experts"
        assert b["repeated_experts"][0]["same"]["unique_experts"] == 4
        assert b["repeated_experts"][3]["distinct"]["unique_experts"] == 16
    print("bench_verify test passed")
