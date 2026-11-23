"""Greedy speculative decoding on MLX with per-layer routing logs.

Runs the target + draft pair over a set of prompts. For every verify pass it
records which experts each row picked in each MoE layer, the draft model's
probability for each drafted token, how many drafts were accepted, and the
verify latency. Optionally dumps full expert inputs/outputs for a few rounds
in the CUDA replay format.

Example (Qwen pair, as in our task list):
  python spec_decode.py --target mlx-community/Qwen3-30B-A3B-4bit \
      --draft mlx-community/Qwen3-0.6B-4bit --num-draft 7 --out runs/qwen

Output: <out>/routing.npz, <out>/summary.json, and <out>/trace/ if --dump-rounds > 0.
"""

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

import common
from common import REC


def draft_step(draft, cache, y):
    """One draft forward on y (1 or 2 tokens). Returns (token, prob)."""
    logits = draft(y[None], cache=cache)[0, -1]
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    tok = mx.argmax(probs)
    return tok, probs[tok]


def speculative_greedy(target, draft, ids, num_draft, max_tokens, on_round=None):
    """Greedy speculative decoding; mirrors mlx_lm's speculative_generate_step.

    on_round(info) is called after every verify pass with a dict holding the
    round's tokens, acceptance, draft probabilities, timing and routing calls.
    Returns the generated token list.
    """
    tcache, dcache = common.new_cache(target), common.new_cache(draft)
    draft_y = common.prefill(draft, dcache, ids)
    y = common.prefill(target, tcache, ids)
    ctx = len(ids) - 1  # tokens already in the target cache

    out = []
    while len(out) < max_tokens:
        k = min(num_draft, max_tokens - len(out))

        # Draft k tokens autoregressively.
        mx.synchronize()
        td = time.perf_counter()
        dtoks, dprobs = [], []
        cur = draft_y
        for _ in range(k):
            t, p = draft_step(draft, dcache, cur)
            mx.async_eval(t, p)
            dtoks.append(t)
            dprobs.append(p)
            cur = t.reshape(1).astype(mx.uint32)
        if k:
            dtok_arr = mx.stack(dtoks).astype(mx.uint32)
            mx.eval(dtok_arr, dprobs)
            verify_in = mx.concatenate([y, dtok_arr])
        else:
            dtok_arr = mx.array([], mx.uint32)
            verify_in = y
        mx.synchronize()
        draft_s = time.perf_counter() - td

        # Verify all k+1 rows in one target pass, recording routing.
        REC.enabled = True
        mx.synchronize()
        t0 = time.perf_counter()
        logits = target(verify_in[None], cache=tcache)[0]
        ttoks = mx.argmax(logits, axis=-1)
        calls = REC.take()  # evaluates routing (and tensors) with the logits
        mx.eval(ttoks)
        mx.synchronize()
        verify_s = time.perf_counter() - t0
        REC.enabled = False

        d = dtok_arr.tolist()
        t = ttoks.tolist()
        n = 0
        while n < k and t[n] == d[n]:
            n += 1
        accepted = d[:n] + [t[n]]
        accepted = accepted[: max_tokens - len(out)]
        out.extend(accepted)

        if on_round is not None:
            on_round(
                dict(
                    ctx_len=ctx,
                    k=k,
                    n_accept=n,
                    draft_tokens=d,
                    target_tokens=t,
                    draft_probs=[float(p.item()) for p in dprobs],
                    verify_s=verify_s,
                    draft_s=draft_s,
                    calls=calls,
                )
            )

        # Rewind exactly as mlx_lm does: the target keeps y + n drafts; the
        # draft never processed its last drafted token.
        common.trim(tcache, k - n)
        common.trim(dcache, max(k - n - 1, 0))
        ctx += n + 1
        y = mx.array([t[n]], mx.uint32)
        draft_y = y
        if n == k and k > 0:
            draft_y = mx.concatenate([mx.array(d[-1:], mx.uint32), y])
    return out


def greedy_baseline(target, ids, max_tokens):
    """Plain autoregressive greedy decoding, for tokens/s and output checking."""
    cache = common.new_cache(target)
    y = common.prefill(target, cache, ids)
    out = []
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        tok = mx.argmax(target(y[None], cache=cache)[0, -1])
        mx.eval(tok)
        out.append(tok.item())
        y = tok.reshape(1).astype(mx.uint32)
    mx.synchronize()
    return out, time.perf_counter() - t0


def routing_array(calls, num_layers, top_k):
    """calls -> int16 [rows, layers, top_k]."""
    rows = calls[0][1].reshape(-1, top_k).shape[0]
    arr = np.full((rows, num_layers, top_k), -1, np.int16)
    for layer, idx, _, _ in calls:
        arr[:, layer, :] = np.array(idx.reshape(-1, top_k)).astype(np.int16)
    if (arr < 0).any():
        raise RuntimeError("some MoE layers were not recorded in a verify pass")
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--num-draft", type=int, default=7, help="drafts per round (rows = this + 1)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=len(common.PROMPTS))
    ap.add_argument("--no-chat", action="store_true", help="skip the chat template")
    ap.add_argument("--baseline", action="store_true", help="also time plain greedy decoding and compare outputs")
    ap.add_argument("--dump-rounds", type=int, default=0, help="rounds to dump in full for the CUDA replay")
    ap.add_argument("--dump-layers", default="auto",
                    help="MoE layers to dump: comma list, 'all', or 'auto' (first, middle, last; "
                         "all 48 layers of Qwen3-30B-A3B would be ~18 GB of weights)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target, draft, tok = common.load_pair(args.target, args.draft)
    run(target, draft, tok, args)


def run(target, draft, tok, args, prompts=None):
    """Separated from main() so tests can call it with tiny models."""
    layers = common.tag_moe_layers(target)
    facts = common.moe_facts(target)
    top_k = common.top_k_of(target)
    L = facts["num_moe_layers"]
    print(f"target: {L} MoE layers, {facts['num_experts']} experts, top-{top_k}, "
          f"{facts['bytes_per_expert'][0] / 2**20:.2f} MiB per expert")

    if args.dump_layers == "all":
        dump_layers = None
    elif args.dump_layers == "auto":  # first, middle, last: keeps the trace small
        dump_layers = {0, L // 2, L - 1}
    else:
        dump_layers = {int(s) for s in args.dump_layers.split(",")}
    writer = None
    if args.dump_rounds > 0:
        from trace_format import TraceWriter

        writer = TraceWriter(os.path.join(args.out, "trace"))
        writer.write_model(target, facts, top_k, layers=None if dump_layers is None else sorted(dump_layers))

    prompts = prompts if prompts is not None else [
        common.encode_prompt(tok, p, chat=not args.no_chat) for p in common.PROMPTS[: args.prompts]
    ]

    routing, offsets = [], [0]
    meta = {key: [] for key in ("prompt_id", "ctx_len", "k", "n_accept", "verify_ms", "draft_ms")}
    dprob_rows = []
    per_prompt = []
    round_id = 0

    for pid, ids in enumerate(prompts):
        def on_round(info):
            nonlocal round_id
            arr = routing_array(info["calls"], L, top_k)
            routing.append(arr)
            offsets.append(offsets[-1] + arr.shape[0])
            meta["prompt_id"].append(pid)
            meta["ctx_len"].append(info["ctx_len"])
            meta["k"].append(info["k"])
            meta["n_accept"].append(info["n_accept"])
            meta["verify_ms"].append(info["verify_s"] * 1e3)
            meta["draft_ms"].append(info["draft_s"] * 1e3)
            dprob_rows.append(info["draft_probs"])
            if writer is not None and len(writer.rounds) < args.dump_rounds:
                writer.write_round(round_id, info["calls"], layers=dump_layers,
                                   modules={i: m for i, _, m in layers})
                if len(writer.rounds) >= args.dump_rounds:
                    REC.capture_tensors = False
            round_id += 1

        REC.capture_tensors = writer is not None and len(writer.rounds) < args.dump_rounds
        t0 = time.perf_counter()
        n_rounds_before = len(meta["k"])
        out = speculative_greedy(target, draft, ids, args.num_draft, args.max_tokens, on_round)
        wall = time.perf_counter() - t0
        rounds = len(meta["k"]) - n_rounds_before
        rec = dict(prompt_id=pid, tokens=len(out), rounds=rounds,
                   spec_tok_per_s=len(out) / wall, tokens_per_round=len(out) / max(rounds, 1))

        if args.baseline:
            base, base_s = greedy_baseline(target, ids, len(out))
            first_diff = next((i for i, (a, b) in enumerate(zip(out, base)) if a != b), None)
            rec.update(base_tok_per_s=len(base) / base_s, speedup=(len(out) / wall) / (len(base) / base_s),
                       matches_greedy=first_diff is None, first_divergence=first_diff)
        per_prompt.append(rec)
        msg = f"prompt {pid}: {len(out)} tokens, {rounds} rounds, {rec['tokens_per_round']:.2f} tok/round, {rec['spec_tok_per_s']:.1f} tok/s"
        if args.baseline:
            msg += f" (greedy {rec['base_tok_per_s']:.1f} tok/s, x{rec['speedup']:.2f}, match={rec['matches_greedy']})"
        print(msg)

    K = max(len(r) for r in dprob_rows) if dprob_rows else 0
    dprobs = np.full((len(dprob_rows), K), np.nan, np.float32)
    for i, r in enumerate(dprob_rows):
        dprobs[i, : len(r)] = r

    np.savez_compressed(
        os.path.join(args.out, "routing.npz"),
        routing=np.concatenate(routing),
        row_offsets=np.array(offsets, np.int64),
        draft_probs=dprobs,
        **{k: np.array(v) for k, v in meta.items()},
    )
    summary = dict(
        target=getattr(args, "target", None), draft=getattr(args, "draft", None),
        num_draft=args.num_draft, max_tokens=args.max_tokens, top_k=top_k, facts=facts,
        rounds=len(meta["k"]), mean_accept=float(np.mean(meta["n_accept"])),
        per_prompt=per_prompt,
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if writer is not None:
        writer.close(extra=dict(source=summary["target"], decoder_layer_index=" ".join(map(str, facts["decoder_layer_index"]))))
    print(f"{summary['rounds']} verify rounds, mean accepted drafts {summary['mean_accept']:.2f} -> {args.out}")
    common.untag(target)
    return summary


if __name__ == "__main__":
    main()
