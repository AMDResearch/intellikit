# Metrix

**GPU Profiling. Decoded.**

Clean, human-readable metrics for AMD GPUs. No more cryptic hardware counters.

## Why Metrix?

Existing GPU profilers are **trash**:
- Cryptic hardware counters everywhere
- No clear interpretation
- Poor software quality
- No unit tests

**Metrix** is different:
- **Clean Python API** with modern design
- **Human-readable metrics** instead of raw counters
- **Unit tested** and reliable
- **12 Memory Metrics**: Bandwidth, cache, coalescing, LDS, atomic latency
- **7 Compute Metrics**: FLOPS, arithmetic intensity (HBM/L2/L1), compute throughput
- **Multi-Run Profiling**: Automatic aggregation with min/max/avg statistics
- **Kernel Filtering**: Efficient regex filtering at rocprofv3 level
- **Multiple Output Formats**: Text, JSON, CSV

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Profile with all metrics (architecture auto-detected)
metrix ./my_app

# Time only (fast)
metrix --time-only -n 10 ./my_app

# Filter kernels by name
metrix --kernel matmul ./my_app

# Custom metrics
metrix --metrics memory.l2_hit_rate,memory.coalescing_efficiency ./my_app

# Save to JSON
metrix -o results.json ./my_app
```

## Python API

```python
from metrix.api import Metrix

# Architecture is auto-detected
profiler = Metrix()
results = profiler.profile("./my_app", num_replays=5)

for kernel in results.kernels:
    print(f"{kernel.name}: {kernel.duration_us.avg:.2f} μs")
    for metric, stats in kernel.metrics.items():
        print(f"  {metric}: {stats.avg:.2f}")
```

## Available Metrics

### Memory Bandwidth
- `memory.hbm_read_bandwidth` - HBM read bandwidth (GB/s)
- `memory.hbm_write_bandwidth` - HBM write bandwidth (GB/s)
- `memory.hbm_bandwidth_utilization` - % of peak HBM bandwidth
- `memory.bytes_transferred_hbm` - Total bytes through HBM
- `memory.bytes_transferred_l2` - Total bytes through L2 cache
- `memory.bytes_transferred_l1` - Total bytes through L1 cache

### Cache Performance
- `memory.l1_hit_rate` - L1 cache hit rate (%)
- `memory.l2_hit_rate` - L2 cache hit rate (%)
- `memory.l2_bandwidth` - L2 cache bandwidth (GB/s)

### Memory Access Patterns
- `memory.coalescing_efficiency` - Memory coalescing efficiency (%)
- `memory.global_load_efficiency` - Global load efficiency (%)
- `memory.global_store_efficiency` - Global store efficiency (%)

### Local Data Share
- `memory.lds_bank_conflicts` - LDS bank conflicts per instruction

### Atomic Operations
- `memory.atomic_latency` - Atomic operation latency (cycles)

### Compute Metrics
- `compute.total_flops` - Total floating-point operations performed
- `compute.hbm_gflops` - Compute throughput (GFLOPS)
- `compute.hbm_arithmetic_intensity` - Ratio of FLOPs to HBM bytes (FLOP/byte)
- `compute.l2_arithmetic_intensity` - Ratio of FLOPs to L2 bytes (FLOP/byte)
- `compute.l1_arithmetic_intensity` - Ratio of FLOPs to L1 bytes (FLOP/byte)

## CLI Options

```
metrix [options] <command>

Options:
  --profile, -p      Use pre-defined profile (quick, memory, compute)
  --metrics, -m      Comma-separated list of metrics
  --time-only        Only collect timing
  --kernel, -k       Filter by kernel name substring
  --runs, -n         Number of runs (default: 10)
  --aggregate        Aggregate by kernel name
  --top K            Show top K kernels
  --output, -o       Output file (.json, .csv, .txt)
  --verbose, -v      Verbose output
  --quiet, -q        Quiet mode

Note: GPU architecture is auto-detected using rocminfo
```

## Testing

```bash
python3 -m pytest tests/ -v
```

## Requirements

- Python 3.9+
- ROCm 6.x with rocprofv3
- pandas>=1.5.0

## Architecture

Metrix uses a clean backend architecture where hardware counter names appear **exactly once** as function parameters:

```python
@metric("memory.l2_hit_rate")
def _l2_hit_rate(self, TCC_HIT_sum, TCC_MISS_sum):
    total = TCC_HIT_sum + TCC_MISS_sum
    return (TCC_HIT_sum / total) * 100 if total > 0 else 0.0
```

This **eliminates error-prone mapping dictionaries** and makes the codebase maintainable.

### Auto-Detection

GPU architecture is automatically detected using `rocminfo`. Metrix will detect your GPU (e.g., gfx942 for MI300X) and use the appropriate backend automatically.

This design makes it easy to add new metrics and support new GPU architectures.

## Example

```bash
$ metrix ./examples/01_vector_add/vector_add

================================================================================
Metrix: all metrics (12 total)
Target: ./examples/01_vector_add/vector_add
================================================================================

────────────────────────────────────────────────────────────────────────────────
Dispatch #1: vector_add(float*, float const*, float const*, int)
────────────────────────────────────────────────────────────────────────────────
Duration: 7.29 - 7.29 μs (avg=7.29)

MEMORY BANDWIDTH:
  Total HBM Bytes Transferred                   8400896.00 bytes
  HBM Bandwidth Utilization                           1.34 percent
  HBM Read Bandwidth                                 35.47 GB/s
  HBM Write Bandwidth                                35.36 GB/s

MEMORY_PATTERN:
  Memory Coalescing Efficiency                      100.00 percent
  Global Load Efficiency                             50.00 percent
  Global Store Efficiency                            25.00 percent

CACHE PERFORMANCE:
  L1 Cache Hit Rate                                  66.67 percent
  L2 Cache Bandwidth Utilization                    144.95 percent
  L2 Cache Hit Rate                                  26.72 percent

LOCAL DATA SHARE (LDS):
  LDS Bank Conflicts                                  0.00 conflicts/instruction

================================================================================
Profiled 1 dispatch(es)/kernel(s)
================================================================================
```

## License

MIT
