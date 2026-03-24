#!/usr/bin/env python3
"""Flash Attention benchmark script for Docker integration tests.

Run inside rocm/pytorch container after: pip install flash-attn, kerncap.
Used by tests/integration/test_flash_attn.py.
"""

import time

import torch
from flash_attn import flash_attn_func

batch, seqlen, nheads, headdim = 2, 512, 8, 64
q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")

# Warmup
for _ in range(3):
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()

# Timed run
start = time.perf_counter()
for _ in range(10):
    out = flash_attn_func(q, k, v)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 10 * 1000
print(f"Flash Attention avg: {elapsed:.3f} ms")
