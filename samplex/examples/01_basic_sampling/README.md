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
================================================================================
Samplex Example: Basic PC Sampling
================================================================================

Step 1: Writing kernel to /tmp...
  Wrote: /tmp/samplex_example_bsximdrf/vector_add.hip

Step 2: Compiling kernel...
  Compiled: /tmp/samplex_example_bsximdrf/vector_add

Step 3: PC sampling with Samplex...

================================================================================
PC SAMPLING RESULTS
================================================================================

Method:     stochastic
Interval:   65536 cycles
Samples:    629
Dispatches: 100

Kernel: vector_add(float const*, float const*, float*, int)
----------------------------------------------------------------------
  Samples:   629
  Duration:  32225.2 us
  Full mask: 100.0%
  Issued:    0.8%
  Top instructions:
     60.7%    382  s_waitcnt  [issued=1, stalled=381]
     28.0%    176  global_load_dword  [issued=4, stalled=172]
      8.4%     53  global_store_dword  [issued=0, stalled=53]
      0.6%      4  v_ashrrev_i32_e32  [issued=0, stalled=4]
      0.6%      4  s_and_saveexec_b64  [issued=0, stalled=4]
      0.5%      3  v_lshl_add_u64  [issued=0, stalled=3]
      0.3%      2  v_add_u32_e32  [issued=0, stalled=2]
      0.3%      2  s_cbranch_execz  [issued=0, stalled=2]
      0.3%      2  s_load_dword  [issued=0, stalled=2]
      0.2%      1  s_load_dwordx4  [issued=0, stalled=1]

================================================================================
```

This vector_add kernel is **memory-bound**: 60.7% of samples are on `s_waitcnt` (waiting
for memory), 28.0% on `global_load_dword` (loading data), and the top stall reason is
WAITCNT at 61.2%. Only 0.8% of samples were issued (actively computing).

See [samplex documentation](../../README.md) for more details.
