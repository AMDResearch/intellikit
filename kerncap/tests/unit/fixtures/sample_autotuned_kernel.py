import triton
import triton.language as tl

autotune_configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=autotune_configs,
    key=["N"],
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(Q, K, V, Out, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < N
    q = tl.load(Q + offs, mask=mask)
    tl.store(Out + offs, q, mask=mask)


@triton.jit
def _helper_kernel(x_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x * 2, mask=mask)
