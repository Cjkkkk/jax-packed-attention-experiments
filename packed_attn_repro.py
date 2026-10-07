"""Synthetic repro: cuDNN packed attention, 2 batch elements x multiple segments.

Compares (a) one packed call over B=2, (b) two B=1 calls, (c) an fp32
per-segment reference, for outputs and dQ/dK/dV with a shared random upstream
gradient.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.cudnn.fused_attention_stablehlo import (
    MaskType, dot_product_attention)

NAMES = ["out", "dQ", "dK", "dV"]


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--heads", type=int, default=16)
  p.add_argument("--hdim", type=int, default=128)
  p.add_argument("--T", type=int, default=8192)
  p.add_argument("--mask", default="no_mask", choices=["no_mask", "causal"])
  p.add_argument("--reps", type=int, default=5)
  p.add_argument("--seg0", default="3072,5120")
  p.add_argument("--seg1", default="2048,6144")
  p.add_argument("--scale", type=float, default=None)
  return p.parse_args()


def make_seqlen_offsets(segs, max_segs):
  """seqlen [B, M] and offsets [B, M+1], padded with -1 (JAX SDPA API)."""
  seqlen = -np.ones((len(segs), max_segs), np.int32)
  offsets = -np.ones((len(segs), max_segs + 1), np.int32)
  for b, s in enumerate(segs):
    seqlen[b, :len(s)] = s
    offsets[b, :len(s)] = np.concatenate([[0], np.cumsum(s)[:-1]])
  return jnp.asarray(seqlen), jnp.asarray(offsets)


def main():
  a = parse_args()
  segs = [[int(x) for x in a.seg0.split(",")],
          [int(x) for x in a.seg1.split(",")]]
  B, T, N, H = len(segs), a.T, a.heads, a.hdim
  M = max(map(len, segs))
  scale = a.scale or H ** -0.5
  causal = a.mask == "causal"
  mask_type = MaskType.CAUSAL if causal else MaskType.NO_MASK

  def sdpa(q, k, v, seqlen, offsets):
    return dot_product_attention(
        q, k, v, q_seqlen=seqlen, kv_seqlen=seqlen, q_offsets=offsets,
        kv_offsets=offsets, scale=scale, mask_type=mask_type,
        qkv_layout="BTNH")

  @jax.jit
  def cudnn_fwd_bwd(q, k, v, g, seqlen, offsets):
    out, vjp = jax.vjp(lambda q, k, v: sdpa(q, k, v, seqlen, offsets), q, k, v)
    return (out, *vjp(g))

  def reference(q, k, v):
    """fp32 attention computed separately for every segment."""
    outs = []
    for b, batch_segs in enumerate(segs):
      start, parts = 0, []
      for length in batch_segs:
        qs, ks, vs = (x[b, start:start + length] for x in (q, k, v))
        logits = jnp.einsum("tnh,snh->nts", qs, ks) * scale
        if causal:
          keep = jnp.tril(jnp.ones((length, length), bool))
          logits = jnp.where(keep, logits, -jnp.inf)
        parts.append(
            jnp.einsum("nts,snh->tnh", jax.nn.softmax(logits, -1), vs))
        start += length
      outs.append(jnp.concatenate(parts, 0))
    return jnp.stack(outs)

  def rel_err(x, ref, valid):
    """Max abs error over valid tokens, relative to max |ref|."""
    x = np.asarray(x, np.float32) * valid
    ref = np.asarray(ref, np.float32) * valid
    return float(np.abs(x - ref).max() / (np.abs(ref).max() + 1e-9))

  kq, kk, kv, kg = jax.random.split(jax.random.key(0), 4)
  q, k, v, g = (
      jax.random.normal(key, (B, T, N, H), jnp.float32).astype(jnp.bfloat16)
      for key in (kq, kk, kv, kg))
  valid = np.zeros((B, T, 1, 1), np.float32)
  for b, s in enumerate(segs):
    valid[b, :sum(s)] = 1

  seqlen, offsets = make_seqlen_offsets(segs, M)
  f32 = lambda *xs: [x.astype(jnp.float32) for x in xs]
  out, vjp = jax.vjp(reference, *f32(q, k, v))
  ref = (out, *vjp(f32(g)[0]))

  print(f"segs={segs} B={B} T={T} N={N} H={H} mask={a.mask}")
  for rep in range(a.reps):
    packed = cudnn_fwd_bwd(q, k, v, g, seqlen, offsets)
    per_batch = []
    for b in range(B):
      sl, of = make_seqlen_offsets([segs[b]], M)
      per_batch.append(cudnn_fwd_bwd(
          q[b:b + 1], k[b:b + 1], v[b:b + 1], g[b:b + 1], sl, of))
    sep = [jnp.concatenate([r[i] for r in per_batch]) for i in range(4)]
    for i, name in enumerate(NAMES):
      print(f"rep{rep} {name}: "
            f"packed-vs-ref {rel_err(packed[i], ref[i], valid):.2e}  "
            f"separate-vs-ref {rel_err(sep[i], ref[i], valid):.2e}  "
            f"packed-vs-sep {rel_err(packed[i], sep[i], valid):.2e}")


if __name__ == "__main__":
  main()
