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

See [samplex documentation](../../README.md) for more details.
