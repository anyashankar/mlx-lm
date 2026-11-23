"""Stock speculative decoding vs pre-pass truncation in mlx-lm, tokens/s and acceptance.

  python moe418/run_prepass.py --model <target> --draft <draft> [--costs granite|qwen] [--tau 0.1]
"""
import argparse, statistics, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, speculative_generate_step
from prepass import GRANITE_VERIFY_MS, QWEN_VERIFY_MS, cost_aware_cut, threshold_stop

PROMPTS = [
    "Explain how a CPU cache works, step by step.",
    "Write a short story about a lighthouse keeper.",
    "Summarize the causes of the French Revolution.",
    "Describe how to make bread from scratch.",
    "What are the tradeoffs between TCP and UDP?",
    "Explain gradient descent to a high school student.",
    "List ten facts about the Moon and explain each one.",
    "Write a Python function that merges two sorted lists, then explain it.",
]


def run(gen, n):
    t = time.perf_counter(); k = 0
    for k, _ in enumerate(gen, 1):
        if k >= n:
            break
    return k / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--num-draft", type=int, default=7); ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--costs", choices=["granite", "qwen"], default="granite")
    ap.add_argument("--tau", type=float, default=0.1)
    a = ap.parse_args()
    model, tok = load(a.model); draft, _ = load(a.draft)
    costs = GRANITE_VERIFY_MS if a.costs == "granite" else QWEN_VERIFY_MS
    configs = {
        "plain": None,
        "spec, all drafts": {},
        "spec, cost-aware cut": dict(draft_cut=cost_aware_cut(costs)),
        f"spec, stop tau={a.tau} + cut": dict(draft_stop=threshold_stop(a.tau), draft_cut=cost_aware_cut(costs)),
    }
    rates = {c: [] for c in configs}
    logs = {c: [] for c in configs}
    for i, text in enumerate(PROMPTS):
        ids = mx.array(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True))
        for c, kw in configs.items():
            if kw is None:
                g = generate_step(ids, model, max_tokens=a.max_tokens)
            else:
                g = speculative_generate_step(ids, model, draft, num_draft_tokens=a.num_draft,
                                              max_tokens=a.max_tokens, round_log=logs[c], **kw)
            r = run(g, a.max_tokens)
            if i > 0:                       # first prompt is warm-up
                rates[c].append(r)
    print(f"| config | tokens/s (median over {len(PROMPTS) - 1} prompts) | drafts made / round | verified / round | accepted / round |")
    print("|---|---|---|---|---|")
    for c in configs:
        L = logs[c]
        f = lambda key: f"{statistics.mean(x[key] for x in L):.2f}" if L else "-"
        print(f"| {c} | {statistics.median(rates[c]):.1f} | {f('drafted')} | {f('verified')} | {f('accepted')} |")


if __name__ == "__main__":
    main()
