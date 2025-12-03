# Example 01: Kernel Tracing

Demonstrates using Nexus to extract assembly and source code from GPU kernels.

## What it does

The Python script:
1. Writes a vector addition kernel to `/tmp`
2. Compiles it with `hipcc -g` (debug symbols required)
3. Uses Nexus to intercept kernel launch and extract code

## Run it

```bash
python3 trace.py
```

## Requirements

- ROCm and `hipcc` installed
- Nexus installed: `pip install git+https://github.com/AMDResearch/nexus.git#egg=intellikit[nexus]`

See [nexus documentation](../../README.md) for more details.

