# Example 01: Basic GPU Kernel Profiling

Demonstrates using Metrix to profile a GPU kernel.

## What it does

The Python script:
1. Writes a vector addition kernel to `/tmp`
2. Compiles it with `hipcc`
3. Profiles it with Metrix (or runs it if profiling not available)

## Run it

```bash
python3 profile.py
```

## Requirements

- ROCm and `hipcc` installed
- Metrix installed: `pip install git+https://github.com/AMDResearch/nexus.git#egg=intellikit[metrix]`

See [metrix documentation](../../README.md) for more details.
