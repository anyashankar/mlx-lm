"""Tiny random MoE models for testing the scripts without downloading weights.

Draft = target with slightly perturbed weights, so drafts are often but not
always accepted, which exercises both the accept and reject paths.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.models import granitemoe, qwen3_moe

VOCAB = 256


def tiny_qwen3_moe(seed=0, layers=4, experts=16, top_k=4, hidden=128, inter=64, quantize=True, dtype=None):
    mx.random.seed(seed)
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=hidden, num_hidden_layers=layers,
        intermediate_size=256, num_attention_heads=4, num_experts=experts,
        num_experts_per_tok=top_k, decoder_sparse_step=1, mlp_only_layers=[],
        moe_intermediate_size=inter, rms_norm_eps=1e-6, vocab_size=VOCAB,
        num_key_value_heads=2, head_dim=32, rope_theta=10000.0,
        tie_word_embeddings=False, max_position_embeddings=4096, norm_topk_prob=True,
    )
    m = qwen3_moe.Model(args)
    if dtype is not None:  # real checkpoints are bf16 before quantization
        m.set_dtype(dtype)
    if quantize:
        nn.quantize(m, group_size=64, bits=4)
    mx.eval(m.parameters())
    return m


def tiny_granite_moe_mlxlm(seed=1, layers=4, experts=10, top_k=4, hidden=128, inter=64):
    """Granite quantized exactly as mlx_lm.convert does: bf16 weights, then
    mlx-lm's Granite recipe (expert down_proj, o_proj, router at 8 bits)."""
    from mlx_lm.utils import quantize_model

    mx.random.seed(seed)
    args = granitemoe.ModelArgs(
        model_type="granitemoe", hidden_size=hidden, num_hidden_layers=layers,
        intermediate_size=inter, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=VOCAB, logits_scaling=1.0, attention_multiplier=1 / 32,
        embedding_multiplier=1.0, residual_multiplier=1.0, max_position_embeddings=4096,
        num_key_value_heads=2, attention_bias=False, rope_theta=10000.0,
        num_local_experts=experts, num_experts_per_tok=top_k,
    )
    m = granitemoe.Model(args)
    m.set_dtype(mx.bfloat16)
    m, _ = quantize_model(m, {}, group_size=64, bits=4, quant_predicate=m.quant_predicate)
    mx.eval(m.parameters())
    return m


def tiny_granite_moe(seed=1, layers=2, experts=8, top_k=2, hidden=128, inter=64):
    mx.random.seed(seed)
    args = granitemoe.ModelArgs(
        model_type="granitemoe", hidden_size=hidden, num_hidden_layers=layers,
        intermediate_size=inter, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=VOCAB, logits_scaling=1.0, attention_multiplier=1 / 32,
        embedding_multiplier=1.0, residual_multiplier=1.0, max_position_embeddings=4096,
        num_key_value_heads=2, attention_bias=False, rope_theta=10000.0,
        num_local_experts=experts, num_experts_per_tok=top_k,
    )
    m = granitemoe.Model(args)
    nn.quantize(m, group_size=64, bits=4)
    mx.eval(m.parameters())
    return m


def perturbed_copy(model_fn, scale=0.02, seed=7, **kw):
    """A copy of model_fn(**kw) with small noise on float weights (a 'draft')."""
    m = model_fn(**kw)
    mx.random.seed(seed)
    params = tree_flatten(m.parameters())
    new = []
    for k, v in params:
        if v.dtype in (mx.float32, mx.float16, mx.bfloat16) and "scales" not in k and "biases" not in k:
            v = v + scale * mx.random.normal(v.shape).astype(v.dtype)
        new.append((k, v))
    m.update(tree_unflatten(new))
    mx.eval(m.parameters())
    return m


def random_prompts(n, length=24, seed=3):
    mx.random.seed(seed)
    return [mx.random.randint(0, VOCAB, (length,)).tolist() for _ in range(n)]
