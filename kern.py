"""Run packed cuDNN attention fwd+bwd only, for kernel-selection inspection (see kn.sh).

Configured via environment variables:
  N, NKV, H  query heads, kv heads (default N), head dim
  SEGS       python literal, per-batch segment lengths
  EXTRA      python literal, extra kwargs for dot_product_attention
Pass "causal" as argv[1] for a causal mask.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.cudnn.fused_attention_stablehlo import (
    MaskType, dot_product_attention)

T = 8192
N = int(os.environ.get("N", 16))
N_KV = int(os.environ.get("NKV", N))
H = int(os.environ.get("H", 128))
SEGS = eval(os.environ.get("SEGS", "[[3072, 5120], [2048, 6144]]"))
EXTRA = eval(os.environ.get("EXTRA", "{}"))
B = len(SEGS)
M = max(map(len, SEGS))

seqlen = -np.ones((B, M), np.int32)
offsets = -np.ones((B, M + 1), np.int32)
for b, segs in enumerate(SEGS):
  seqlen[b, :len(segs)] = segs
  offsets[b, :len(segs)] = np.concatenate([[0], np.cumsum(segs)[:-1]])
seqlen, offsets = jnp.asarray(seqlen), jnp.asarray(offsets)

mask_type = (MaskType.CAUSAL if sys.argv[1:] == ["causal"]
             else MaskType.NO_MASK)


def attention(q, k, v):
  return dot_product_attention(
      q, k, v, q_seqlen=seqlen, kv_seqlen=seqlen, q_offsets=offsets,
      kv_offsets=offsets, scale=0.1, mask_type=mask_type, qkv_layout="BTNH",
      **EXTRA)


@jax.jit
def fwd_bwd(q, k, v):
  out, vjp = jax.vjp(attention, q, k, v)
  return out, vjp(out)


def normal(seed, heads):
  return jax.random.normal(jax.random.key(seed), (B, T, heads, H), jnp.bfloat16)


q, k, v = normal(0, N), normal(1, N_KV), normal(2, N_KV)
for _ in range(3):
  jax.block_until_ready(fwd_bwd(q, k, v))
print("done")
