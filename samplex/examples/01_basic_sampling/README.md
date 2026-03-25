# Example 01: Basic PC Sampling

Demonstrates using Samplex to profile a GPU kernel and find instruction-level hotspots.

## What it does

The Python script:
1. Writes a vector addition kernel to `/tmp`
2. Compiles it with `hipcc`
3. Profiles it with Samplex using stochastic PC sampling

## Run it

```bash
python3 sample.py
```

## Requirements

- ROCm and `hipcc` installed
- MI300+ GPU (gfx942 or later) for stochastic sampling
- Samplex installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=samplex"`

## Expected Output

```
Samplex PC Sampling Results (stochastic, interval=65536 cycles)
======================================================================
Command:    /tmp/samplex_example_bsximdrf/vector_add
Method:     stochastic
Samples:    629
Dispatches: 100

Global Instruction Breakdown
----------------------------------------------------------------------
   60.7%     382  s_waitcnt
   28.0%     176  global_load_dword
    8.4%      53  global_store_dword
    0.6%       4  v_ashrrev_i32_e32
    0.6%       4  s_and_saveexec_b64
    0.5%       3  v_lshl_add_u64
    0.3%       2  v_add_u32_e32
    0.3%       2  s_cbranch_execz
    0.3%       2  s_load_dword
    0.2%       1  s_load_dwordx4

Kernel: vector_add(float const*, float const*, float*, int)
----------------------------------------------------------------------
  Samples:     629
  Duration:    32225.2 us
  Full mask:   100.0%
  Issued:      0.8%
  Stall reasons:
     61.2%  WAITCNT
     30.9%  ARBITER_NOT_WIN
      5.8%  ARBITER_WIN_EX_STALL
      1.4%  ALU_DEPENDENCY
      0.6%  NO_INSTRUCTION_AVAILABLE
  Top instructions:
     60.7%    382  s_waitcnt vmcnt(0) [issued=1, stalled=381]
     28.0%    176  global_load_dword v6, v[4:5], off [issued=4, stalled=172]
      8.4%     53  global_store_dword v[0:1], v2, off [issued=0, stalled=53]
      0.6%      4  v_ashrrev_i32_e32 v1, 31, v0 [issued=0, stalled=4]
      0.6%      4  s_and_saveexec_b64 s[2:3], vcc [issued=0, stalled=4]
      0.5%      3  v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1] [issued=0, stalled=3]
      0.3%      2  v_add_u32_e32 v0, s2, v0 [issued=0, stalled=2]
      0.3%      2  s_cbranch_execz 22 [issued=0, stalled=2]
      0.3%      2  s_load_dword s3, s[0:1], 0x2c [issued=0, stalled=2]
      0.2%      1  s_load_dwordx4 s[4:7], s[0:1], 0x0 [issued=0, stalled=1]
```

This vector_add kernel is **memory-bound**: 60.7% of samples are on `s_waitcnt` (waiting
for memory), 28.0% on `global_load_dword` (loading data), and the top stall reason is
WAITCNT at 61.2%. Only 0.8% of samples were issued (actively computing).

See [samplex documentation](../../README.md) for more details.
