"""Write traces from tiny models, build the CPU replay, and require it to pass;
then corrupt one expert id and require it to fail.

Two traces: an all-4-bit Qwen3-MoE, and a Granite-MoE quantized with mlx-lm's
own recipe (expert down_proj at 8 bits), which is what mlx_lm.convert produces."""

import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]

import mlx.core as mx
import numpy as np

import common
import spec_decode
from tiny_models import perturbed_copy, random_prompts, tiny_granite_moe_mlxlm, tiny_qwen3_moe

mx.set_default_device(mx.cpu)


def check(name, target, draft, exe, d):
    out = os.path.join(d, name)
    spec_decode.run(target, draft, None, SimpleNamespace(num_draft=7, max_tokens=24, baseline=False, dump_rounds=3,
                    dump_layers="auto", out=out, target=name, draft=name), prompts=random_prompts(1))
    trace = os.path.join(out, "trace")
    r = subprocess.run([exe, trace], capture_output=True, text=True)
    print(r.stdout.splitlines()[0])
    print("\n".join(r.stdout.splitlines()[-4:-2]))
    assert r.returncode == 0, f"{name}: replay failed on a clean trace\n{r.stdout}\n{r.stderr}"

    bad = os.path.join(d, name + "_bad")
    shutil.copytree(trace, bad)
    layer = sorted(int(f.split("_L")[1].split("_")[0]) for f in os.listdir(os.path.join(bad, "rounds")))[-1]
    p = os.path.join(bad, "rounds", f"R1_L{layer}_idx.npy")
    a = np.load(p)
    E = common.moe_facts(target)["num_experts"]
    a[0, 0] = (a[0, 0] + 1) % E
    np.save(p, a)
    r = subprocess.run([exe, bad], capture_output=True, text=True)
    assert r.returncode == 1 and "FAIL" in r.stdout, f"{name}: replay did not catch a corrupted expert id"
    print(f"ok: {name} clean trace passes, corrupted trace fails")


if __name__ == "__main__":
    cuda_dir = os.path.join(HERE, "..", "cuda")
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "replay_cpu")
        subprocess.run(["g++", "-O2", "-std=c++17", "-Wno-unknown-pragmas", "-x", "c++",
                        os.path.join(cuda_dir, "replay.cu"), "-o", exe], check=True)

        q = tiny_qwen3_moe(layers=6, dtype=mx.bfloat16)
        qd = perturbed_copy(tiny_qwen3_moe, scale=0.01, layers=6, dtype=mx.bfloat16)
        check("qwen-4bit", q, qd, exe, d)

        g = tiny_granite_moe_mlxlm()
        assert common.moe_facts(g)["proj_bits"] == {"gate": 4, "up": 4, "down": 8}, common.moe_facts(g)["proj_bits"]
        check("granite-mixed-4-8bit", g, g, exe, d)
    print("trace replay test passed")
