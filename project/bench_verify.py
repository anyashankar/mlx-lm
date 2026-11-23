"""Verify latency at 1-8 rows, plus MoE-only timing and a repeated-expert test.

Three measurements, all on the target model:

1. Full verify pass at r = 1..max-rows rows on real contexts. Rows are the
   target's own greedy continuation (what fully accepted drafts look like), so
   routing is realistic rather than random. Each r is timed several times in
   shuffled order and the median kept; one extra untimed pass records how many
   distinct experts each layer read.

2. MoE-only time: just the expert matmuls (every layer's SwitchGLU) on the
   recorded inputs and expert ids, i.e. "MoE time per round" without
   attention, router or LM head.

3. Repeated expert ids: one MoE layer timed with synthetic routing where every
   row picks the SAME top-k experts vs. rows picking DIFFERENT experts. Same
   number of (row, expert) pairs, very different number of distinct experts.
   If "same" costs as much as "distinct", MLX's gather_qmm re-reads repeated
   experts from memory; if "same" stays near the 1-row cost, it does not.
   Note mlx-lm sorts indices once rows*top_k >= 64 (8 rows at top-8), which
   changes the access pattern; the output marks those points.

Example:
  python bench_verify.py --target mlx-community/Qwen3-30B-A3B-4bit --out runs/qwen
"""

import argparse
import json
import os
import random
import statistics

import mlx.core as mx
import numpy as np

import common
from common import REC


def median_time(fn, reps, warmup=2):
    for _ in range(warmup):
        common.sync_time(fn)
    ts = [common.sync_time(fn)[0] for _ in range(reps)]
    return statistics.median(ts), ts


def verify_latency(target, contexts, max_rows, reps, L, top_k):
    """contexts: list of (prompt_ids, continuation_ids). Returns list of records."""
    records = []
    for cid, (ids, cont) in enumerate(contexts):
        cache = common.new_cache(target)
        y = common.prefill(target, cache, ids)
        rows_all = mx.concatenate([y, mx.array(cont, mx.uint32)])

        def fwd(r):
            def f():
                out = target(rows_all[:r][None], cache=cache)
                return out
            return f

        # warm up shapes, then interleave row counts so drift hits all equally
        for r in range(1, max_rows + 1):
            common.sync_time(fwd(r))
            common.trim(cache, r)
        times = {r: [] for r in range(1, max_rows + 1)}
        order = [r for r in range(1, max_rows + 1) for _ in range(reps)]
        random.Random(cid).shuffle(order)
        for r in order:
            t, _ = common.sync_time(fwd(r))
            common.trim(cache, r)
            times[r].append(t)

        # one recorded, untimed pass per r for distinct experts and inputs
        for r in range(1, max_rows + 1):
            REC.enabled, REC.capture_tensors = True, True
            target(rows_all[:r][None], cache=cache)
            calls = REC.take()
            REC.enabled, REC.capture_tensors = False, False
            common.trim(cache, r)
            ids_lr = np.zeros((L, r, top_k), np.int32)
            for layer, idx, _, _ in calls:
                ids_lr[layer] = np.array(idx.reshape(-1, top_k))
            unique = [int(len(np.unique(ids_lr[l]))) for l in range(L)]
            records.append(dict(
                context=cid, rows=r, ctx_len=len(ids) - 1,
                verify_ms_median=statistics.median(times[r]) * 1e3,
                verify_ms_all=[t * 1e3 for t in times[r]],
                unique_experts_per_layer=unique,
                routing=ids_lr.tolist(),
                _calls=calls,
            ))
    return records


def moe_only_time(target, calls, reps):
    """Time every MoE layer's SwitchGLU on recorded (x, ids); returns seconds."""
    mods = {i: m for i, _, m in common.find_moe_layers(target)}
    REC.enabled = False

    def f():
        return [mods[layer](x, idx) for layer, idx, x, _ in calls]

    t, _ = median_time(f, reps)
    return t


def repeated_expert_test(target, layer, max_rows, top_k, reps, bytes_per_expert):
    """Same-experts vs distinct-experts routing on one layer."""
    _, _, mod = common.find_moe_layers(target)[layer]
    E = mod.gate_proj.weight.shape[0]
    hidden = common.moe_facts(target)["hidden"]
    dtype = mod.gate_proj.scales.dtype if hasattr(mod.gate_proj, "scales") else mx.float32
    out = []
    for r in range(1, max_rows + 1):
        x = mx.random.normal((1, r, hidden)).astype(dtype)
        same = mx.broadcast_to(mx.arange(top_k, dtype=mx.uint32), (1, r, top_k))
        distinct = (mx.arange(r * top_k, dtype=mx.uint32) % E).reshape(1, r, top_k)
        rec = dict(rows=r, pairs=r * top_k, sorted_path=r * top_k >= 64)
        for name, idx in (("same", same), ("distinct", distinct)):
            idx = mx.array(idx)
            t, _ = median_time(lambda: mod(x, idx), reps)
            u = int(len(np.unique(np.array(idx))))
            rec[name] = dict(ms=t * 1e3, unique_experts=u,
                             unique_GBps=u * bytes_per_expert / t / 1e9,
                             pair_GBps=r * top_k * bytes_per_expert / t / 1e9)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--contexts", type=int, default=6, help="prompts to use as contexts")
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--micro-layer", type=int, default=0, help="MoE layer for the repeated-expert test")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mlx_lm import load

    target, tok = load(args.target)
    prompts = [common.encode_prompt(tok, p, chat=not args.no_chat) for p in common.PROMPTS[: args.contexts]]
    run(target, prompts, args)


def run(target, prompts, args):
    os.makedirs(args.out, exist_ok=True)
    common.tag_moe_layers(target)
    facts = common.moe_facts(target)
    top_k = common.top_k_of(target)
    L = facts["num_moe_layers"]
    bpe = facts["bytes_per_expert"]

    # continuation = target's own greedy output, so rows look like accepted drafts
    from spec_decode import greedy_baseline

    contexts = []
    for ids in prompts:
        cont, _ = greedy_baseline(target, ids, args.max_rows - 1)
        contexts.append((ids, cont))

    print(f"timing full verify pass, rows 1..{args.max_rows}, {len(contexts)} contexts x {args.reps} reps")
    recs = verify_latency(target, contexts, args.max_rows, args.reps, L, top_k)

    print("timing MoE-only (expert matmuls of all layers)")
    for rec in recs:
        rec["moe_ms_median"] = moe_only_time(target, rec.pop("_calls"), args.reps) * 1e3
        rec["experts_read_total"] = sum(rec["unique_experts_per_layer"])
        rec["bytes_read_unique"] = sum(u * b for u, b in zip(rec["unique_experts_per_layer"], bpe))
        rec["bytes_read_pairs"] = sum(rec["rows"] * top_k * b for b in bpe)

    print(f"repeated-expert test on MoE layer {args.micro_layer}")
    rep = repeated_expert_test(target, args.micro_layer, args.max_rows, top_k, args.reps, bpe[args.micro_layer])

    with open(os.path.join(args.out, "bench.json"), "w") as f:
        json.dump(dict(facts=facts, top_k=top_k, max_rows=args.max_rows, reps=args.reps,
                       verify=recs, repeated_experts=rep), f)

    print(f"\n{'rows':>4} {'verify ms':>10} {'MoE ms':>8} {'experts read':>13} {'MiB unique':>11} {'MoE GB/s':>9}")
    for r in range(1, args.max_rows + 1):
        rs = [x for x in recs if x["rows"] == r]
        v = statistics.median(x["verify_ms_median"] for x in rs)
        m = statistics.median(x["moe_ms_median"] for x in rs)
        e = statistics.mean(x["experts_read_total"] for x in rs)
        b = statistics.mean(x["bytes_read_unique"] for x in rs)
        print(f"{r:>4} {v:>10.2f} {m:>8.2f} {e:>13.0f} {b / 2**20:>11.1f} {b / (m / 1e3) / 1e9:>9.1f}")
    print(f"\n{'rows':>4} {'same ms':>9} {'distinct ms':>12} {'distinct uniq':>14}  sorted")
    for x in rep:
        print(f"{x['rows']:>4} {x['same']['ms']:>9.3f} {x['distinct']['ms']:>12.3f} "
              f"{x['distinct']['unique_experts']:>14}  {'yes' if x['sorted_path'] else ''}")
    common.untag(target)
    print(f"-> {os.path.join(args.out, 'bench.json')}")


if __name__ == "__main__":
    main()
