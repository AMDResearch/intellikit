# Metrix

**GPU Profiling. Decoded.**

Clean, human-readable metrics for AMD GPUs. No more cryptic hardware counters.

## Why Metrix?

Existing GPU profilers have challenges:
- Cryptic hardware counters everywhere
- No clear interpretation
- Inconsistent software quality
- Limited testing

**Metrix** takes a different approach:
- **Clean Python API** with modern design
- **Human-readable metrics** instead of raw counters
- **Unit tested** and reliable
- **13 Memory Metrics**: Bandwidth, cache, coalescing, LDS, atomic latency
- **5 Compute Metrics**: FLOPS, arithmetic intensity (HBM/L2/L1), compute throughput
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

## Distributed Launchers

Metrix supports distributed profiling with launchers like `torchrun`, `mpirun`,
`srun`, and `horovodrun`. Use `--launcher` to specify the launcher command
separately from the application, ensuring the correct invocation order
(`launcher rocprofv3 ... -- app`).

```bash
# Torch distributed
metrix profile --launcher "torchrun --nproc_per_node=8" -- train.py

# MPI
metrix profile --launcher "mpirun -np 8" -- ./my_app --problem-size 4096

# Slurm
metrix profile --launcher "srun -N 2 -n 16" -- ./my_app
```

When distributed rank variables are present (`RANK`, `OMPI_COMM_WORLD_RANK`,
`SLURM_PROCID`, etc.), Metrix emits per-rank metadata and automatically suffixes
`--output/-o` files with a rank tag (for example, `results.rank0003.json`) to avoid
cross-rank clobbering.

## Python API

```python
from metrix import Metrix

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

Profiling uses the `profile` subcommand (or omit `profile` when the first argument is your app or a flag — Metrix inserts `profile` for you).

```
metrix [--version] <command> ...

metrix profile [options] <target>

  --profile, -p      Metric profile: quick | memory | memory_bandwidth |
                     memory_cache | compute (default: all metrics if omitted)
  --metrics, -m      Comma-separated list of metrics (mutually exclusive with -p / --time-only)
  --time-only        Only collect timing, no hardware counters
  --kernel, -k       Filter kernels by name (regular expression, passed to rocprofv3)
  --num-replays, -n  Replay the application N times and aggregate (default: 10)
  --aggregate        Aggregate metrics by kernel name across replays (default: per-dispatch across runs)
  --top K            Show only top K slowest kernels
  --output, -o       Output file (.json, .csv, .txt)
  --timeout SECONDS  Profiling timeout in seconds (default: 60)
  --log, -l          Logging level: debug | info | warning | error (default: warning)
  --quiet, -q        Quiet mode
  --no-counters      Omit raw counter values from output

metrix list <metrics|profiles|devices> [--category CAT]

metrix info <metric|profile> <name>

```

`metrix list counters` and `metrix info counter <name>` exist but currently print “not yet implemented” in the CLI.

Note: GPU architecture is auto-detected using `rocminfo`.

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

See the [examples directory](examples/) for complete working examples.

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
