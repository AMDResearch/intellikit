# Proboscis — Agent-Driven GPU Kernel Instrumentation

Instrument GPU kernels in running applications without source code modification.
Uses the hidden-argument ABI trick to inject probes at dispatch time.

## Quick Start

```python
from proboscis import Proboscis

p = Proboscis()
result = p.instrument(["./vec_add"], "find memory accesses")
for kernel in result:
    print(f"{kernel.kernel_name}: {kernel.record_count} records")
    print(kernel.summary)
```

## MCP Server

```bash
proboscis-mcp  # starts the MCP server
```

### Tools

- `instrument_kernel(command, probe, ...)` — Run with instrumentation
- `list_probes()` — Available probe types
- `analyze_results(path)` — Analyze saved results

## Probe Types

- **memory_trace** — "find memory accesses", "trace loads and stores"
- **block_count** — "count basic block executions", "find hotspots"
- **register_snapshot** — "check register pressure", "VGPR usage"

## How It Works

1. Agent describes what to instrument (natural language)
2. Proboscis translates to a ProbeSpec
3. C++ runtime (libproboscis.so) loaded via HSA_TOOLS_LIB
4. Intercepts kernel dispatches, repacks kernarg buffers
5. Hidden probe context pointer added to each kernel's arguments
6. Results collected in GPU-visible probe buffer
7. Structured results returned to the agent
