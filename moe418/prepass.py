"""Pre-pass truncation rules for mlx_lm.generate.speculative_generate_step.

Use with:  speculative_generate_step(..., draft_stop=threshold_stop(0.1),
                                     draft_cut=cost_aware_cut(GRANITE_VERIFY_MS))

The draft probabilities are the draft model's raw probabilities of its own greedy tokens.
They are uncalibrated; the go/no-go replay calibrated them against observed acceptance,
which these rules do not do yet.
"""
from typing import Callable, List, Sequence

# Median verify time (ms) on the M5 Max by number of verified rows, from the milestone report
# (measured at 1, 2, 4, 6, 8 rows; odd counts interpolated linearly).
_GRANITE = {1: 4.56, 2: 5.63, 4: 7.32, 6: 8.77, 8: 9.80}
_QWEN = {1: 12.35, 2: 17.09, 4: 25.80, 6: 35.48, 8: 40.04}


def _fill(m):
    out = []
    for r in range(1, 9):
        if r in m:
            out.append(m[r])
        else:
            out.append((m[r - 1] + m[r + 1]) / 2)
    return out


GRANITE_VERIFY_MS = _fill(_GRANITE)   # index r-1 = cost of verifying r rows
QWEN_VERIFY_MS = _fill(_QWEN)


def threshold_stop(tau: float) -> Callable[[List[float]], bool]:
    """Stop drafting once the chance that every draft so far is accepted drops below tau."""
    def stop(p: List[float]) -> bool:
        acc = 1.0
        for x in p:
            acc *= x
        return acc < tau
    return stop


def cost_aware_cut(verify_ms: Sequence[float]) -> Callable[[List[float]], int]:
    """Verify the k drafts that maximize expected tokens per verify ms.

    With k drafts the target checks k + 1 rows and emits 1 + sum_{i<=k} prod_{j<=i} p_j tokens
    in expectation (the bonus token plus each accepted prefix). Draft time is already spent.
    """
    def cut(p: List[float]) -> int:
        best_k, best, acc, exp_tokens = 1, -1.0, 1.0, 1.0
        for k in range(1, len(p) + 1):
            acc *= p[k - 1]
            exp_tokens += acc
            rows = min(k + 1, len(verify_ms))
            rate = exp_tokens / verify_ms[rows - 1]
            if rate > best:
                best, best_k = rate, k
        return best_k
    return cut
