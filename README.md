# Packed-attention backward kernel / B=2 multi-segment correctness (synthetic)

Files: packed_attn_repro.py (correctness; packed B=2 vs 2x B=1 vs fp32 per-segment ref, out/dQ/dK/dV),
kern.py (kernel selection), kn.sh (nsys kernel names, needs --cuda-graph-trace=node), run.sh.
JAX 0.11.2 + jax-cuda13 installed to ./pylib (container's /opt/jax jaxlib/plugin PJRT API mismatch segfaults at GPU init).
cudnn_<ver>/ dirs: alternative cuDNN wheels; use PYTHONPATH=$PWD/cudnn_<ver>:$PWD/pylib.

Results on B200 (SM100), BF16, B=2, T=8192, N=16, H=128, segs [[3072,5120],[2048,6144]]:
- cuDNN 9.19.0, 9.21.0, 9.23.2, 9.25.1, 9.27.0: fwd=sdpa_sm100_flash_fprop, bwd=sdpa_sm100_flash_bprop;
  packed == separate == fp32 ref (rel err ~3e-3 bf16 noise; packed-vs-separate dQ ~1e-3), stable over repeats, no_mask and causal.
- SM100 bprop also selected for: H=64/256, GQA 32/8, sliding window, unaligned segments, 4 segs/element.
- NOT reproduced: wmma bwd selection, or B=2 corruption.
