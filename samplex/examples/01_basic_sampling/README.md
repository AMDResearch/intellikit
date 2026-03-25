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
  Wrote: /tmp/samplex_example_3gff9jbk/vector_add.hip

Step 2: Compiling kernel...
  Compiled: /tmp/samplex_example_3gff9jbk/vector_add

Step 3: PC sampling with Samplex...

================================================================================
PC SAMPLING RESULTS
================================================================================

Method:     stochastic
Interval:   65536 cycles
Samples:    600
Dispatches: 100

Kernel: vector_add(float const*, float const*, float*, int)
----------------------------------------------------------------------
  Samples:   600
  Duration:  404.2 us
  Full mask: 100.0%
  Issued:    9.7%
  Top instructions:
     70.5%    423  s_waitcnt  [issued=0, stalled=423]
      4.2%     25  global_load_dword  [issued=5, stalled=20]
      3.5%     21  v_lshl_add_u64  [issued=11, stalled=10]
      3.5%     21  s_load_dword  [issued=1, stalled=20]
      3.3%     20  s_and_saveexec_b64  [issued=3, stalled=17]
      3.0%     18  v_ashrrev_i32_e32  [issued=3, stalled=15]
      3.0%     18  global_store_dword  [issued=3, stalled=15]
      2.0%     12  v_add_u32_e32  [issued=3, stalled=9]
      1.8%     11  v_lshlrev_b64  [issued=5, stalled=6]
      1.3%      8  v_add_f32_e32  [issued=8, stalled=0]

================================================================================
```

This vector_add kernel is **memory-bound**: 70.5% of samples are on `s_waitcnt` (waiting
for memory), and most instructions show high stalled counts relative to issued. Only 9.7%
of samples were issued (actively computing).

See [samplex documentation](../../README.md) for more details.
