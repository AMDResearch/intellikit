# Example 01: Vector Addition

## Overview
Simple vector addition kernel demonstrating basic memory profiling capabilities.

**What it does:** `c[i] = a[i] + b[i]` for 1M elements (4MB per array, 12MB total)

**What to learn:**
- How to profile memory bandwidth
- Understanding L2 cache hit rates
- Interpreting coalesced memory access
- Reading min/max/avg statistics
- Multiple runs vs. single run profiling

## Build and Run

```bash
make
./vector_add
```

Expected output:
```
Vector Addition Example
=======================
Array size: 1048576 elements (4.00 MB per array)
Total data: 12.00 MB

Launching kernel: grid=4096, block=256
Result: ✓ PASS
```

## Profile with Metrix

### 1. Basic Profiling (Timing Only)

```bash
metrix ./vector_add --time-only
```

**Expected Output:**
```
================================================================================
Metrix: timing-only
Target: ./vector_add
================================================================================

================================================================================
Individual Dispatches (1 total)
================================================================================

────────────────────────────────────────────────────────────────────────────────
Dispatch #1: vector_add(float*, float const*, float const*, int)
────────────────────────────────────────────────────────────────────────────────
GPU: 1
Duration: 5.89 μs
Grid: (1048576, 1, 1), Workgroup: (256, 1, 1)

================================================================================
Total: 1 dispatch(es)
================================================================================
```

**Note:** Default behavior shows individual dispatches. Each dispatch is shown separately.

### 2. Multiple Runs with Statistics

```bash
metrix ./vector_add --runs 5 --aggregate --time-only
```

**Expected Output:**
```
================================================================================
Metrix: timing-only
Target: ./vector_add
Runs: 5
================================================================================

────────────────────────────────────────────────────────────────────────────────
Kernel: vector_add(float*, float const*, float const*, int)
────────────────────────────────────────────────────────────────────────────────
Dispatches: 5
Duration:   6.45 μs  (min=5.89, max=7.21)

================================================================================
Profiled 1 kernel(s)
================================================================================
```

**Note:** With `--aggregate`, dispatches are grouped by kernel name, showing min/max/avg statistics.

### 3. Memory Profile (Default)

```bash
metrix ./vector_add
```

**Expected Output:**
```
================================================================================
Metrix: profile 'memory'
Target: ./vector_add
================================================================================

================================================================================
Individual Dispatches (1 total)
================================================================================

────────────────────────────────────────────────────────────────────────────────
Dispatch #1: vector_add(float*, float const*, float const*, int)
────────────────────────────────────────────────────────────────────────────────
GPU: 1
Duration: 5.89 μs
Grid: (1048576, 1, 1), Workgroup: (256, 1, 1)

Metrics:
  Total HBM Bytes Transferred              8400896.00 bytes
  L1 Cache Hit Rate                             66.67 percent
  L2 Cache Hit Rate                             26.72 percent
  Memory Coalescing Efficiency                 100.00 percent
  Global Load Efficiency                       100.00 percent
  Global Store Efficiency                      100.00 percent
  LDS Bank Conflicts                             0.00 conflicts/instruction

================================================================================
Total: 1 dispatch(es)
================================================================================
```

### 4. Memory Profile with Aggregation

```bash
metrix ./vector_add --runs 5 --aggregate
```

**Expected Output:**
```
================================================================================
Metrix: profile 'memory'
Target: ./vector_add
Runs: 5
================================================================================

────────────────────────────────────────────────────────────────────────────────
Kernel: vector_add(float*, float const*, float const*, int)
────────────────────────────────────────────────────────────────────────────────
Dispatches: 5
Duration:   6.45 μs  (min=5.89, max=7.21)

────────────────────────────────────────────────────────────────────────────────
MEMORY BANDWIDTH
────────────────────────────────────────────────────────────────────────────────
  Total HBM Bytes Transferred         8400896.00 bytes      (min=8400896.00, max=8400896.00)

────────────────────────────────────────────────────────────────────────────────
CACHE PERFORMANCE
────────────────────────────────────────────────────────────────────────────────
  L1 Cache Hit Rate                      66.67 percent    (min=66.67, max=66.67)
  L2 Cache Hit Rate                      26.72 percent    (min=26.71, max=26.73)

────────────────────────────────────────────────────────────────────────────────
MEMORY ACCESS PATTERNS
────────────────────────────────────────────────────────────────────────────────
  Global Store Efficiency               100.00 percent    (min=100.00, max=100.00)
  Memory Coalescing Efficiency          100.00 percent    (min=100.00, max=100.00)
  Global Load Efficiency                100.00 percent    (min=100.00, max=100.00)

────────────────────────────────────────────────────────────────────────────────
LOCAL DATA SHARE (LDS)
────────────────────────────────────────────────────────────────────────────────
  LDS Bank Conflicts                      0.00 conflicts/instruction (min=0.00, max=0.00)

================================================================================
Profiled 1 kernel(s)
================================================================================
```

**Note:** With `--aggregate`, metrics are organized into clean sections with min/max/avg statistics.

### 5. Specific Metrics

```bash
metrix ./vector_add --metrics memory.l2_hit_rate
```

**Expected Output:**
```
================================================================================
Individual Dispatches (1 total)
================================================================================

────────────────────────────────────────────────────────────────────────────────
Dispatch #1: vector_add(float*, float const*, float const*, int)
────────────────────────────────────────────────────────────────────────────────
GPU: 1
Duration: 5.89 μs
Grid: (1048576, 1, 1), Workgroup: (256, 1, 1)

Metrics:
  L2 Cache Hit Rate                             26.72 percent

================================================================================
Total: 1 dispatch(es)
================================================================================
```

### 6. JSON Output

```bash
metrix ./vector_add --format json --output results.json
```

Machine-readable JSON format for post-processing and visualization.

## Python API Usage

```python
from metrix import Metrix

# Create profiler
profiler = Metrix(arch="gfx942")

# Profile with specific metrics
results = profiler.profile(
    command="./vector_add",
    metrics=["memory.hbm_bandwidth_utilization", "memory.l2_hit_rate"],
    timeout=60
)

# Access results
for kernel in results.kernels:
    print(f"{kernel.name}:")
    print(f"  Duration: {kernel.duration.avg:.2f} μs")
    print(f"  L2 Hit Rate: {kernel.metrics['memory.l2_hit_rate'].avg:.1f}%")
```

## Key Takeaways

✅ **Default: Individual dispatches** - See each dispatch separately
✅ **--aggregate for statistics** - Group by kernel, show min/max/avg
✅ **--runs N for multiple runs** - Run the app N times for better statistics
✅ **Always reports timing** - Even without counter collection
✅ **Units shown everywhere** - μs, ms, %, GB/s - never ambiguous
✅ **Logical counter names** - Architecture-independent (memory.read_requests, not TCC_EA0_RDREQ_sum)
✅ **Clean sectioned output** - MEMORY BANDWIDTH, CACHE PERFORMANCE, etc.

## Performance Characteristics

- **Memory Bound**: Yes - limited by memory bandwidth
- **Compute Intensity**: Very low (just addition)
- **Cache Behavior**: Streaming - no reuse expected
- **Coalescing**: Perfect - consecutive threads access consecutive memory

## Next Steps

- Try varying array sizes to see cache effects
- Compare to Example 02 (cache-friendly patterns)
- Benchmark on different GPU architectures
