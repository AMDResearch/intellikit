---
myst:
    html_meta:
        "description": "Walk through the full IntelliKit workflow: profile a GPU application, inspect execution, optimize a kernel, and validate correctness using Metrix, Nexus, Linex, Omniprobe, Kerncap, and Accordo."
        "keywords": "IntelliKit, GPU workflow, ROCm, HIP, Metrix, Nexus, Linex, Omniprobe, Kerncap, Accordo, kernel optimization"
---

# Use IntelliKit end to end

This topic explains the full IntelliKit workflow: profiling a GPU application, inspecting execution, optimizing a kernel, and validating correctness.

## The pipeline

The IntelliKit workflow follows five stages.

```
Isolate → Profile → Inspect → Optimize → Validate
```

1. **Isolate** a kernel with Kerncap
2. **Profile** it with Metrix (hardware counters), Linex (source-line timing), and Omniprobe (memory access patterns)
3. **Inspect** execution details with Nexus (assembly + HIP source)
4. **Optimize** the kernel in isolation
5. **Validate** correctness with Accordo

## Step 1: Profile and identify the target kernel

Use Metrix to find the hot kernels and understand their performance characteristics.

```python
from metrix import Metrix

profiler = Metrix()
results = profiler.profile(
    "./my_app",
    metrics=["memory.hbm_bandwidth_utilization"],
)

for kernel in results.kernels:
    bw = kernel.metrics["memory.hbm_bandwidth_utilization"].avg
    print(f"{kernel.name}: {kernel.avg_time_us:.2f} μs/dispatch, BW util: {bw:.1f}%")
```

## Step 2: Inspect GPU execution details

Use Nexus to inspect what ran on the GPU, including the assembly and HIP source for each kernel.

```python
from nexus import Nexus

trace = Nexus().run(["./my_app"])
for kernel in trace:
    print(f"{kernel.name}: {len(kernel.assembly)} instructions")
```

## Step 3: Get source-line profiling

Use Linex to map kernel performance to specific source lines. Compile your application with `-g` for source-line mapping.

```python
from linex import Linex

profiler = Linex()
profiler.profile("./my_app", kernel_filter="target_kernel")

for line in profiler.source_lines[:5]:
    print(f"{line.file}:{line.line_number}")
    print(f"  {line.total_cycles:,} cycles ({line.stall_percent:.1f}% stalled)")
```

### Find the accesses wasting bandwidth

Linex tells you which lines are slow. Omniprobe tells you which accesses are moving more data than
they need to, and how much that costs. Compile with `-g` here as well — without it the counts are
still correct, but they cannot be mapped back to source.

```python
from omniprobe_ik import Omniprobe

probe = Omniprobe()
result = probe.analyze_memory("./my_app", kernel_filter="target_kernel")

for kernel in result.kernels:
    worst = sorted(kernel.accesses, key=lambda a: a.excess_cache_lines, reverse=True)
    for access in worst[:5]:
        print(f"{access.source_file}:{access.line}:{access.column}")
        print(f"  {access.excess_cache_lines} excess cache lines, "
              f"{access.bank_conflicts} bank conflicts")
```

Omniprobe also reports per-basic-block cycle counts as a distribution through
`analyze_basic_blocks`, so a block whose `p99` sits far above its `p50` stands out as one being
reached under very different conditions — something an average hides.

```{note}
Omniprobe drives a separate tool of the same name. The `omniprobe` executable must be on `PATH`
and its Python API importable; see [Install IntelliKit](../getting-started/installation.md).
```

## Step 4: Isolate and optimize the kernel

Use Kerncap to extract the kernel into a standalone reproducer, then iterate on it.

```{note}
This step requires `compile_commands.json` from your project's CMake build. Generate it by configuring CMake with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. Without it, `make recompile` is unavailable. See [Kerncap reference](../tools/kerncap.md) for details.
```

```python
from kerncap import Kerncap
import subprocess, os

kc = Kerncap()

# Extract the kernel
result = kc.extract(
    "target_kernel",
    cmd=["./my_app"],
    source_dir="./src",
    output="./isolated/target_kernel",
)
reproducer_dir = result.output_dir

# Edit kernel_variant.cpp, then recompile and benchmark
subprocess.run(["make", "recompile"], cwd=reproducer_dir, check=True)

baseline = kc.replay(reproducer_dir)
variant = kc.replay(reproducer_dir, hsaco=os.path.join(reproducer_dir, "optimized.hsaco"))
print(f"Speedup: {baseline.timing_us / variant.timing_us:.2f}x")
```

## Step 5: Validate kernel correctness

Use Accordo to ensure the optimized kernel produces correct results.

```python
from accordo import Accordo

validator = Accordo(binary="./my_app", kernel_name="target_kernel")
ref = validator.capture_snapshot(binary="./my_app")
opt = validator.capture_snapshot(binary="./my_app_opt")
result = validator.compare_snapshots(ref, opt, tolerance=1e-6)

if result.is_valid:
    print(f"PASS — {result.num_arrays_validated} arrays matched")
else:
    print(result.summary())
```

## End-to-end workflow 

Here is a complete IntelliKit workflow for profiling, inspecting, and validating your GPU kernel optimizations with Metrix, Nexus, and Accordo:

```python
from metrix import Metrix
from nexus import Nexus
from accordo import Accordo

# 1) Baseline metrics
profiler = Metrix()
baseline = profiler.profile(
    "./app_baseline",
    metrics=["memory.hbm_bandwidth_utilization"],
)
baseline_bw = baseline.kernels[0].metrics["memory.hbm_bandwidth_utilization"].avg

# 2) See what ran on the GPU
trace = Nexus().run(["./app_baseline"])
for kernel in trace:
    print(kernel.name, len(kernel.assembly), "instructions")

# 3) After you optimize — check correctness
validator = Accordo(binary="./app_baseline", kernel_name="my_kernel")
ref = validator.capture_snapshot(binary="./app_baseline")
opt = validator.capture_snapshot(binary="./app_opt")
result = validator.compare_snapshots(ref, opt, tolerance=1e-6)

if result.is_valid:
    opt_results = profiler.profile(
        "./app_opt",
        metrics=["memory.hbm_bandwidth_utilization"],
    )
    opt_bw = opt_results.kernels[0].metrics["memory.hbm_bandwidth_utilization"].avg
    print(f"PASS — {result.num_arrays_validated} arrays matched; BW delta {opt_bw - baseline_bw:.1f}%")
```
