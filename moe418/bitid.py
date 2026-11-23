"""Bit-identity test: are the kept rows of a truncated verify pass identical to the same rows of the full pass?

For each context, the target runs the next R tokens as one verify pass (R rows) on top of a cached prefix,
then the first k rows alone for k = 1..R-1, and compares logits row for row. Repeats the full pass to separate
run-to-run nondeterminism from batch-shape effects.

  python moe418/bitid.py --model <target> [--rows 8] [--contexts 6] [--dtype model|float32]
"""
import argparse
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

TEXT = ("The history of computing is a story of abstraction. Early machines were programmed by rewiring "
        "panels, then by punched cards, then by assembly language, and eventually by high level languages "
        "that hid the hardware entirely. Each step let programmers think less about the machine and more "
        "about the problem. Caches, pipelines and parallel processors changed what fast code looks like, "
        "but compilers and libraries absorbed much of that complexity. Today a single laptop holds a GPU "
        "with thousands of lanes, and large language models run on it by streaming billions of weights "
        "from memory for every token they produce. ") * 4


def pass_logits(model, cache, toks):
    out = model(mx.array(toks)[None], cache=cache)
    mx.eval(out)
    trim_prompt_cache(cache, len(toks))
    return out[0].astype(mx.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--contexts", type=int, default=6)
    ap.add_argument("--dtype", choices=["model", "float32"], default="model")
    a = ap.parse_args()
    model, tok = load(a.model)
    if a.dtype == "float32":
        model.set_dtype(mx.float32)
    ids = tok.encode(TEXT)
    R = a.rows
    print(f"model {a.model}, activations {a.dtype}, {R}-row verify pass vs first k rows alone\n")
    print("| context | k | rows identical | max abs logit diff | argmax agrees | full pass repeat identical |")
    print("|---|---|---|---|---|---|")
    totals = {"rows": 0, "identical": 0, "argmax": 0}
    for c in range(a.contexts):
        start = 64 + c * (len(ids) - 64 - R) // max(a.contexts, 1)
        ctx, rows = ids[:start], ids[start:start + R]
        cache = make_prompt_cache(model)
        model(mx.array(ctx)[None], cache=cache)
        mx.eval([x.state for x in cache])
        full = pass_logits(model, cache, rows)
        again = pass_logits(model, cache, rows)
        repeat_ok = bool(mx.array_equal(full, again).item())
        for k in range(1, R):
            part = pass_logits(model, cache, rows[:k])
            ref = full[:k]
            same = [bool(mx.array_equal(ref[i], part[i]).item()) for i in range(k)]
            diff = mx.abs(ref - part).max().item()
            agree = int((ref.argmax(-1) == part.argmax(-1)).sum().item())
            totals["rows"] += k; totals["identical"] += sum(same); totals["argmax"] += agree
            print(f"| {c} | {k} | {sum(same)}/{k} | {diff:.3g} | {agree}/{k} | {repeat_ok} |")
    t = totals
    print(f"\ntotal: {t['identical']}/{t['rows']} kept rows bit-identical, {t['argmax']}/{t['rows']} same argmax")


if __name__ == "__main__":
    main()
