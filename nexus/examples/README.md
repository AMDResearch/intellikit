# Nexus Examples

Simple examples demonstrating how to use Nexus to trace GPU kernels.

## Examples

### `simple_hip.py`

Traces a simple HIP vector addition kernel.

**Run:**
```bash
python3 examples/simple_hip.py
```

**What it does:**
- Creates a simple HIP add kernel in a temp file
- Compiles and runs it with Nexus tracing
- Shows captured assembly and HIP source

### `simple_triton.py`

Traces a Triton kernel.

**Run:**
```bash
python3 examples/simple_triton.py
```

**What it does:**
- Creates a Triton add kernel in a temp file
- Runs it with Nexus tracing
- Shows captured assembly and source

### `multiple_hip.py`

Traces multiple HIP kernels in a single execution.

**Run:**
```bash
python3 examples/multiple_hip.py
```

**What it does:**
- Creates two HIP kernels (add and multiply)
- Compiles and runs them with Nexus tracing
- Shows assembly and source for both kernels

### `multiple_triton.py`

Traces multiple Triton kernels in a single execution.

**Run:**
```bash
python3 examples/multiple_triton.py
```

**What it does:**
- Creates two Triton kernels (add and multiply)
- Runs both with Nexus tracing
- Shows assembly and source for both kernels

## Usage Pattern

Both examples follow the same simple pattern:

```python
from nexus import Nexus

# Create tracer
nexus = Nexus(log_level=1)

# Run and get trace
trace = nexus.run(["python", "my_script.py"])

# Analyze kernels
for kernel in trace:
    print(f"{kernel.name}: {len(kernel.assembly)} instructions")
    print(kernel.hip)
```

## Prerequisites

- ROCm installed
- For HIP example: `hipcc` in PATH
- For Triton example: `triton` installed (`pip install triton`)
- Nexus installed: `pip install git+https://github.com/AMDResearch/nexus.git`
