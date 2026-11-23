"""Expert-union statistics, memory traffic, cost table, and the go/no-go.

Inputs:  <run>/routing.npz + <run>/summary.json  (from spec_decode.py)
         <run>/bench.json                         (from bench_verify.py)
Outputs: <run>/analysis.json, <run>/report.md, <run>/fig_*.png

THE GO/NO-GO QUESTION
In-pass truncation only beats a pre-pass rule (EVICT-style) if the experts a
round actually reads differ enough from the average that knowing them changes
the cut, AND the first few layers' routing predicts the rest. This script
replays every logged round under several cut policies, using the measured
cost model, and compares realized tokens per millisecond:

  fixed-all      verify every drafted row
  fixed-best     the single best fixed row count, chosen in hindsight
  pre-pass       per round, maximize E[tokens]/avg_cost(r), using draft
                 probabilities and the AVERAGE cost of r rows (what EVICT does)
  in-pass@l      decide after seeing routing for the first l MoE layers;
                 earlier layers already ran with all rows; later layers'
                 experts are extrapolated from the first l
  in-pass-oracle same, but with this round's full routing known (upper bound)

E[tokens](r) = 1 + sum_{i<r} prod_{j<=i} q_j. By default q_j is the draft
probability calibrated to how often drafts with that probability were really
accepted (fit 2-fold by prompt); --value draft uses raw draft probabilities.
Every policy uses the same estimate, so only the cost information differs.
Realized tokens for r rows = min(accepted, r-1) + 1.
Cost(r) = a + b * experts_read(r) + c * r  (fit from bench.json) + draft time.

Verdict: GO if the best realistic in-pass policy beats pre-pass by at least
--threshold (default 5%) in tokens/ms. If only the oracle clears it, the
per-round information is there but early layers do not predict it. If even
the oracle does not, in-pass truncation cannot beat a pre-pass rule here.
Simplifications: one cut point per round (the real kernel can cut at every
layer), and attention for dropped rows is charged in proportion to layers.
"""

import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------
def load_run(d):
    z = np.load(os.path.join(d, "routing.npz"))
    summary = json.load(open(os.path.join(d, "summary.json")))
    bench_p = os.path.join(d, "bench.json")
    bench = json.load(open(bench_p)) if os.path.exists(bench_p) else None
    return z, summary, bench


def prefix_unions(routing, offsets, E):
    """Per round: U[r-1, l] = distinct experts among the first r rows at layer l."""
    out = []
    for i in range(len(offsets) - 1):
        R = routing[offsets[i]: offsets[i + 1]].astype(np.int64)  # rows, L, k
        rows, L, k = R.shape
        mask = np.zeros((rows, L, E), bool)
        np.put_along_axis(mask, R, True, axis=2)
        out.append(np.logical_or.accumulate(mask, axis=0).sum(-1))  # rows, L
    return out


def expected_tokens(p, r):
    """1 + sum_{i=1}^{r-1} prod_{j<=i} p_j."""
    if r <= 1:
        return 1.0
    return 1.0 + float(np.cumprod(p[: r - 1]).sum())


def conditional_samples(dprobs, k, n, sel):
    """(p, accepted) for every draft that was actually judged: draft j counts
    only if drafts 1..j-1 were accepted (greedy verify stops at the first miss)."""
    ps, ys = [], []
    for i in np.flatnonzero(sel):
        for j in range(min(int(n[i]) + 1, int(k[i]))):
            ps.append(dprobs[i, j])
            ys.append(1.0 if n[i] >= j + 1 else 0.0)
    return np.array(ps), np.array(ys)


def fit_calibration(ps, ys, bins=10, prior=5.0):
    """Binned map p -> P(accepted | earlier drafts accepted), with quantile
    bin edges and shrinkage toward the overall rate (prior pseudo-counts)."""
    base = ys.mean() if len(ys) else 0.5
    if len(ps) < 2 * bins:
        return lambda p: np.full_like(np.asarray(p, float), base)
    edges = np.unique(np.quantile(ps, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, ps, side="right") - 1, 0, len(edges) - 2)
    rate = np.array([(ys[idx == b].sum() + prior * base) / ((idx == b).sum() + prior)
                     for b in range(len(edges) - 1)])

    def q(p):
        b = np.clip(np.searchsorted(edges, np.asarray(p, float), side="right") - 1, 0, len(edges) - 2)
        return rate[b]
    return q


def value_model(dprobs, k, n, prompt_id, mode="calibrated"):
    """Per-round arrays q_j used in E[tokens](r) = 1 + sum_i prod_{j<=i} q_j.

    'draft'      q_j = the draft model's own probability (what EVICT uses)
    'calibrated' q_j = P(accepted | earlier accepted, draft prob), learned
                 2-fold by prompt so no round is scored by a map fit on itself
    """
    if mode == "draft":
        return np.nan_to_num(dprobs), {"mode": "draft"}
    folds = prompt_id % 2
    q = np.zeros_like(dprobs, dtype=float)
    held_p, held_y, held_q = [], [], []
    single = len(np.unique(folds)) < 2
    for f in np.unique(folds):
        train = np.ones(len(k), bool) if single else folds != f
        cal = fit_calibration(*conditional_samples(dprobs, k, n, train))
        test = folds == f
        q[test] = cal(np.nan_to_num(dprobs[test]))
        p_, y_ = conditional_samples(dprobs, k, n, test)
        held_p += list(p_)
        held_y += list(y_)
        held_q += list(cal(p_))
    held_y, held_q = np.array(held_y), np.array(held_q)
    info = {"mode": "calibrated", "two_fold": not single,
            "heldout_brier": float(((held_q - held_y) ** 2).mean()) if len(held_y) else None,
            "heldout_brier_raw": float(((np.array(held_p) - held_y) ** 2).mean()) if len(held_y) else None}
    return q, info


def fit_cost(bench):
    """verify_ms = a + b*experts_read + c*rows, by least squares."""
    v = bench["verify"]
    X = np.array([[1.0, x["experts_read_total"], x["rows"]] for x in v])
    y = np.array([x["verify_ms_median"] for x in v])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    Xm = X[:, :2]
    ym = np.array([x["moe_ms_median"] for x in v])
    cm, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    return dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), r2=float(r2),
                moe_a=float(cm[0]), moe_b=float(cm[1]))


# --------------------------------------------------------------------------
def union_stats(U_list, top_k, bpe):
    L = U_list[0].shape[1]
    max_rows = max(U.shape[0] for U in U_list)
    rows_stats = []
    for r in range(1, max_rows + 1):
        tot = np.array([U[r - 1].sum() for U in U_list if U.shape[0] >= r])
        per_layer = np.array([U[r - 1].mean() for U in U_list if U.shape[0] >= r])
        new = (np.array([(U[r - 1] - (U[r - 2] if r > 1 else 0)).sum() for U in U_list if U.shape[0] >= r]))
        uniq_bytes = np.array([(U[r - 1] * bpe).sum() for U in U_list if U.shape[0] >= r])
        pair_bytes = r * top_k * bpe.sum()
        rows_stats.append(dict(
            rows=r, rounds=int(len(tot)),
            experts_total_mean=float(tot.mean()), experts_total_std=float(tot.std()),
            experts_total_cv=float(tot.std() / tot.mean()),
            experts_total_p10=float(np.percentile(tot, 10)), experts_total_p90=float(np.percentile(tot, 90)),
            per_layer_mean=float(per_layer.mean()), no_overlap_per_layer=r * top_k,
            overlap_ratio=float(per_layer.mean() / (r * top_k)),
            new_experts_mean=float(new.mean()), new_experts_cv=float(new.std() / max(new.mean(), 1e-9)),
            MiB_unique_mean=float(uniq_bytes.mean() / 2**20),
            MiB_if_repeats_reread=float(pair_bytes / 2**20),
            repeat_waste_frac=float(1 - uniq_bytes.mean() / pair_bytes),
        ))
    return rows_stats


def acceptance_stats(k, n, dprobs):
    K = int(k.max())
    full = k == K
    pos = [float((n[full] >= i).mean()) for i in range(1, K + 1)]
    # calibration: predicted P(first i drafts accepted) vs observed
    bins = np.linspace(0, 1, 6)
    pred, obs = [], []
    for row, kk, nn in zip(dprobs, k, n):
        cp = np.cumprod(row[:kk])
        for i in range(kk):
            pred.append(cp[i])
            obs.append(1.0 if nn >= i + 1 else 0.0)
    pred, obs = np.array(pred), np.array(obs)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (pred >= lo) & (pred < hi if hi < 1 else pred <= hi)
        if sel.any():
            cal.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(sel.sum()),
                            predicted=float(pred[sel].mean()), observed=float(obs[sel].mean())))
    return dict(num_draft=K, mean_accepted=float(n.mean()),
                p_accept_at_least=pos, calibration=cal)


# --------------------------------------------------------------------------
def simulate(U_list, k, n, qvals, cost, draft_ms, decision_layers):
    a, b, c = cost["a"], cost["b"], cost["c"]
    L = U_list[0].shape[1]
    maxr = max(U.shape[0] for U in U_list)
    # average experts read at r rows (what a pre-pass profile knows)
    avg_X = np.array([np.mean([U[r - 1].sum() for U in U_list if U.shape[0] >= r]) for r in range(1, maxr + 1)])

    def tok(i, r):
        return min(int(n[i]), r - 1) + 1

    def cost_full(i, r):
        return a + b * U_list[i][r - 1].sum() + c * r + draft_ms[i]

    policies = {}

    def add(name, choose, cost_fn=cost_full):
        T = C = 0.0
        picks = []
        for i in range(len(U_list)):
            r = choose(i)
            T += tok(i, r)
            C += cost_fn(i, r)
            picks.append(r)
        policies[name] = dict(tokens=T, cost_ms=C, tok_per_ms=T / C, mean_rows=float(np.mean(picks)))

    rows_of = [U.shape[0] for U in U_list]
    add("fixed-all", lambda i: rows_of[i])
    best_fixed = None
    for rr in range(1, maxr + 1):
        add(f"_fixed{rr}", lambda i, rr=rr: min(rr, rows_of[i]))
        if best_fixed is None or policies[f"_fixed{rr}"]["tok_per_ms"] > policies[best_fixed]["tok_per_ms"]:
            best_fixed = f"_fixed{rr}"
    policies["fixed-best"] = dict(policies[best_fixed], rows=int(best_fixed[6:]))
    for key in [p for p in policies if p.startswith("_fixed")]:
        del policies[key]

    def argmax_r(i, cost_r):
        p = qvals[i][: rows_of[i] - 1]
        vals = [expected_tokens(p, r) / cost_r(r) for r in range(1, rows_of[i] + 1)]
        return int(np.argmax(vals)) + 1

    add("pre-pass", lambda i: argmax_r(i, lambda r: a + b * avg_X[r - 1] + c * r + draft_ms[i]))
    add("in-pass-oracle", lambda i: argmax_r(i, lambda r: cost_full(i, r)))

    for lstar in decision_layers:
        frac = lstar / L

        def realized(i, r, lstar=lstar, frac=frac):
            U = U_list[i]
            full = rows_of[i]
            X = U[full - 1, :lstar].sum() + U[r - 1, lstar:].sum()
            return a + b * X + c * (frac * full + (1 - frac) * r) + draft_ms[i]

        def estimated(i, r, lstar=lstar, frac=frac):
            U = U_list[i]
            full = rows_of[i]
            X = U[full - 1, :lstar].sum() + U[r - 1, :lstar].sum() * (L - lstar) / lstar
            return a + b * X + c * (frac * full + (1 - frac) * r) + draft_ms[i]

        add(f"in-pass@{lstar}", lambda i, est=estimated: argmax_r(i, lambda r: est(i, r)), cost_fn=realized)
    return policies


def early_late_correlation(U_list, decision_layers):
    """Does the last row's new-expert count in early layers predict later layers?"""
    K = max(U.shape[0] for U in U_list)
    full = [U for U in U_list if U.shape[0] == K and K > 1]
    if len(full) < 3:
        return {}
    new = np.array([U[K - 1] - U[K - 2] for U in full])  # rounds, L
    out = {}
    L = new.shape[1]
    for lstar in decision_layers:
        early, late = new[:, :lstar].sum(1), new[:, lstar:].sum(1)
        if early.std() == 0 or late.std() == 0:
            out[lstar] = None
        else:
            out[lstar] = float(np.corrcoef(early, late)[0, 1])
    return out


# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(d, rstats, bench, policies, top_k):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return []
    figs = []

    # 1. experts read per layer vs rows, with the no-overlap ceiling
    r = [x["rows"] for x in rstats]
    m = [x["per_layer_mean"] for x in rstats]
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.plot(r, [rr * top_k for rr in r], "--", color=INK2, linewidth=1.2)
    ax.text(r[-1] + 0.2, r[-1] * top_k, "no overlap", color=INK2, fontsize=9, va="center")
    ax.plot(r, m, "-o", color=BLUE, linewidth=2, markersize=6)
    ax.text(r[-1] + 0.2, m[-1], f"measured ({m[-1]:.1f})", color=INK, fontsize=9, va="center")
    ax.set_xlabel("rows in the verify pass (1 + drafts)", color=INK2, fontsize=9)
    ax.set_ylabel("distinct experts per layer", color=INK2, fontsize=9)
    ax.set_title("Experts each verify pass reads, per MoE layer", color=INK, fontsize=11, loc="left")
    ax.set_xlim(0.5, r[-1] + 2.2)
    ax.set_ylim(0, r[-1] * top_k * 1.08)
    fig.tight_layout()
    p = os.path.join(d, "fig_experts_vs_rows.png")
    fig.savefig(p, dpi=200)
    figs.append(p)
    plt.close(fig)

    # 2. verify and MoE latency vs rows (same unit, one axis)
    if bench:
        rows = sorted({x["rows"] for x in bench["verify"]})
        vm = [np.median([x["verify_ms_median"] for x in bench["verify"] if x["rows"] == q]) for q in rows]
        mm = [np.median([x["moe_ms_median"] for x in bench["verify"] if x["rows"] == q]) for q in rows]
        fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
        _style(ax)
        ax.plot(rows, vm, "-o", color=BLUE, linewidth=2, markersize=6, label="full verify pass")
        ax.plot(rows, mm, "-o", color=ORANGE, linewidth=2, markersize=6, label="expert matmuls only")
        ax.text(rows[-1] + 0.2, vm[-1], f"verify {vm[-1] / vm[0]:.2f}x", color=INK, fontsize=9, va="center")
        ax.text(rows[-1] + 0.2, mm[-1], f"MoE {mm[-1] / mm[0]:.2f}x", color=INK, fontsize=9, va="center")
        ax.set_ylim(0, max(vm) * 1.15)
        ax.set_xlim(0.5, rows[-1] + 2.0)
        ax.set_xlabel("rows in the verify pass", color=INK2, fontsize=9)
        ax.set_ylabel("ms (median)", color=INK2, fontsize=9)
        ax.set_title("Verify latency vs rows (labels: cost relative to 1 row)", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
        fig.tight_layout()
        p = os.path.join(d, "fig_latency_vs_rows.png")
        fig.savefig(p, dpi=200)
        figs.append(p)
        plt.close(fig)

    # 3. policies: change in tokens per ms vs pre-pass (signed bars about zero)
    if policies:
        base = policies["pre-pass"]["tok_per_ms"]
        names = list(policies)
        vals = [(policies[k]["tok_per_ms"] / base - 1) * 100 for k in names]
        colors = [AQUA if k.startswith("in-pass") else (BLUE if k == "pre-pass" else INK2) for k in names]
        fig, ax = plt.subplots(figsize=(6, 0.42 * len(names) + 1.2), facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        span = max(max(abs(v) for v in vals) * 1.35, 1.0)
        for yy, v in zip(y, vals):
            ax.text(v + (0.02 if v >= 0 else -0.02) * span, yy, f"{v:+.1f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=INK)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, color=INK, fontsize=9)
        ax.set_xlim(-span, span)
        ax.set_xlabel("change in tokens per ms vs pre-pass (%)", color=INK2, fontsize=9)
        ax.set_title("Replayed rounds under each cut policy", color=INK, fontsize=11, loc="left")
        fig.tight_layout()
        p = os.path.join(d, "fig_policies.png")
        fig.savefig(p, dpi=200)
        figs.append(p)
        plt.close(fig)
    return figs


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="directory with routing.npz, summary.json, bench.json")
    ap.add_argument("--threshold", type=float, default=0.05, help="min tokens/ms gain over pre-pass for GO")
    ap.add_argument("--cost", help="override cost model as a,b,c (ms, ms/expert, ms/row)")
    ap.add_argument("--no-draft-cost", action="store_true", help="leave draft time out of the cost")
    ap.add_argument("--value", choices=["calibrated", "draft"], default="calibrated",
                    help="acceptance estimate used by the cut rules (see value_model)")
    args = ap.parse_args()
    analyze(args.run, args.threshold, args.cost, not args.no_draft_cost, args.value)


def analyze(d, threshold=0.05, cost_override=None, include_draft=True, value="calibrated"):
    z, summary, bench = load_run(d)
    facts = summary["facts"]
    top_k = summary["top_k"]
    E = facts["num_experts"]
    bpe = np.array(facts["bytes_per_expert"])
    L = facts["num_moe_layers"]
    U_list = prefix_unions(z["routing"], z["row_offsets"], E)
    k, n, dprobs = z["k"], z["n_accept"], z["draft_probs"]
    draft_ms = z["draft_ms"] if include_draft and "draft_ms" in z else np.zeros(len(k))

    rstats = union_stats(U_list, top_k, bpe)
    acc = acceptance_stats(k, n, dprobs)
    decision_layers = sorted({x for x in (1, 2, 4, 8, L // 4, L // 2) if 1 <= x < L})
    corr = early_late_correlation(U_list, decision_layers)

    if cost_override:
        a, b, c = (float(v) for v in cost_override.split(","))
        cost = dict(a=a, b=b, c=c, r2=None, source="override")
    elif bench:
        cost = fit_cost(bench)
        cost["source"] = "bench.json"
    else:
        cost = None

    qvals, value_info = value_model(dprobs, k, n, z["prompt_id"], value)
    policies = simulate(U_list, k, n, qvals, cost, draft_ms, decision_layers) if cost else None

    verdict = None
    if policies:
        pre = policies["pre-pass"]["tok_per_ms"]
        gain = {name: v["tok_per_ms"] / pre - 1 for name, v in policies.items()}
        realistic = {k_: g for k_, g in gain.items() if k_.startswith("in-pass@")}
        best_real = max(realistic, key=realistic.get)
        if realistic[best_real] >= threshold:
            verdict = ("GO", f"{best_real} beats pre-pass by {realistic[best_real]:+.1%} (threshold {threshold:.0%}).")
        elif gain["in-pass-oracle"] >= threshold:
            verdict = ("MARGINAL", f"Perfect per-round routing would beat pre-pass by {gain['in-pass-oracle']:+.1%}, "
                       f"but the best realistic cut ({best_real}) only gets {realistic[best_real]:+.1%}: early layers "
                       f"do not predict later traffic well enough.")
        else:
            verdict = ("NO-GO", f"Even perfect per-round routing beats pre-pass by only {gain['in-pass-oracle']:+.1%} "
                       f"(threshold {threshold:.0%}). Expert usage does not vary enough between rounds; "
                       f"take the fallback (cost model + pre-pass rule + kernels as a sync study).")

    result = dict(facts=facts, top_k=top_k, rounds=len(k), acceptance=acc, union_by_rows=rstats,
                  cost_model=cost, value_model=value_info, policies=policies, early_late_corr=corr,
                  verdict=verdict, threshold=threshold)
    if bench:
        result["cost_table"] = cost_table(bench)
        result["repeated_experts"] = bench["repeated_experts"]
    with open(os.path.join(d, "analysis.json"), "w") as f:
        json.dump(result, f, indent=1)
    figs = make_figures(d, rstats, bench, policies, top_k)
    write_report(d, result, figs)
    print(open(os.path.join(d, "report.md")).read())
    return result


def cost_table(bench):
    rows = sorted({x["rows"] for x in bench["verify"]})
    v1 = None
    out = []
    for r in rows:
        rs = [x for x in bench["verify"] if x["rows"] == r]
        v = float(np.median([x["verify_ms_median"] for x in rs]))
        m = float(np.median([x["moe_ms_median"] for x in rs]))
        e = float(np.mean([x["experts_read_total"] for x in rs]))
        bu = float(np.mean([x["bytes_read_unique"] for x in rs]))
        v1 = v1 or v
        out.append(dict(rows=r, verify_ms=v, relative_to_1_row=v / v1, moe_ms=m,
                        experts_read=e, MiB_unique=bu / 2**20, moe_GBps=bu / (m / 1e3) / 1e9))
    return out


def write_report(d, R, figs):
    L = []
    w = L.append
    f = R["facts"]
    w(f"# Routing analysis\n")
    w(f"{f['num_moe_layers']} MoE layers, {f['num_experts']} experts, top-{R['top_k']}, "
      f"{np.mean(f['bytes_per_expert']) / 2**20:.2f} MiB per expert; {R['rounds']} verify rounds.\n")
    if R["verdict"]:
        w(f"## Go/no-go: **{R['verdict'][0]}**\n\n{R['verdict'][1]}\n")
    a = R["acceptance"]
    w("## Acceptance\n")
    w(f"Mean accepted drafts per round: {a['mean_accepted']:.2f} of {a['num_draft']}.\n")
    w("| draft position | P(accepted at least this many) |\n|---|---|")
    for i, p in enumerate(a["p_accept_at_least"], 1):
        w(f"| {i} | {p:.2f} |")
    w("\nDraft-probability calibration (predicted vs observed P(first i drafts accepted)):\n")
    w("| bin | n | predicted | observed |\n|---|---|---|---|")
    for c in a["calibration"]:
        w(f"| {c['bin']} | {c['n']} | {c['predicted']:.2f} | {c['observed']:.2f} |")
    w("\n## Expert-union statistics\n")
    w("| rows | rounds | experts/layer | no-overlap | overlap ratio | total experts (mean ± sd) | CV across rounds | new experts this row | MiB read (unique) | MiB if repeats re-read | waste if re-read |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for x in R["union_by_rows"]:
        w(f"| {x['rows']} | {x['rounds']} | {x['per_layer_mean']:.1f} | {x['no_overlap_per_layer']} | {x['overlap_ratio']:.2f} | "
          f"{x['experts_total_mean']:.0f} ± {x['experts_total_std']:.0f} | {x['experts_total_cv']:.3f} | {x['new_experts_mean']:.1f} | "
          f"{x['MiB_unique_mean']:.0f} | {x['MiB_if_repeats_reread']:.0f} | {x['repeat_waste_frac']:.0%} |")
    if "cost_table" in R:
        w("\n## Measured cost table\n")
        w("| rows | verify ms | vs 1 row | MoE ms | experts read | MiB unique | MoE GB/s |\n|---|---|---|---|---|---|---|")
        for x in R["cost_table"]:
            w(f"| {x['rows']} | {x['verify_ms']:.2f} | {x['relative_to_1_row']:.2f}x | {x['moe_ms']:.2f} | "
              f"{x['experts_read']:.0f} | {x['MiB_unique']:.0f} | {x['moe_GBps']:.0f} |")
        c = R["cost_model"]
        w(f"\nFit: verify_ms = {c['a']:.3f} + {c['b']:.4f} x experts_read + {c['c']:.3f} x rows "
          f"(R^2 = {c['r2']:.3f}).\n")
        w("Repeated expert ids (one layer; same number of row-expert pairs):\n")
        w("| rows | same experts ms | distinct experts ms | distinct unique | sorted path |\n|---|---|---|---|---|")
        for x in R["repeated_experts"]:
            w(f"| {x['rows']} | {x['same']['ms']:.3f} | {x['distinct']['ms']:.3f} | {x['distinct']['unique_experts']} | {'yes' if x['sorted_path'] else ''} |")
        w("\nIf 'same' grows like 'distinct', repeated ids are re-read from memory; if 'same' stays flat, they are not.")
    if R["policies"]:
        w("\n## Cut policies on the logged rounds\n")
        vm = R["value_model"]
        if vm["mode"] == "calibrated":
            w(f"Acceptance estimate: draft probabilities calibrated to observed acceptance"
              f"{' (2-fold by prompt)' if vm['two_fold'] else ' (fit on all rounds: only one prompt fold)'}; "
              f"held-out Brier score {vm['heldout_brier']:.3f} vs {vm['heldout_brier_raw']:.3f} for raw draft probabilities.\n")
        else:
            w("Acceptance estimate: raw draft probabilities (EVICT-style).\n")
        pre = R["policies"]["pre-pass"]["tok_per_ms"]
        w("| policy | tokens/ms | vs pre-pass | mean rows verified |\n|---|---|---|---|")
        for k, v in R["policies"].items():
            w(f"| {k} | {v['tok_per_ms']:.4f} | {v['tok_per_ms'] / pre - 1:+.1%} | {v['mean_rows']:.2f} |")
    if R["early_late_corr"]:
        w("\nDoes early routing predict later traffic? Correlation, across rounds, between the last row's new experts "
          "in the first l layers and in the remaining layers:\n")
        w("| l | correlation |\n|---|---|")
        for k, v in R["early_late_corr"].items():
            w(f"| {k} | {'n/a' if v is None else f'{v:.2f}'} |")
    for p in figs:
        w(f"\n![]({os.path.basename(p)})")
    open(os.path.join(d, "report.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
