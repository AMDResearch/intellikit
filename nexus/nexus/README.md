# Nexus Python Package

Python utilities for tracing and analyzing ROCm GPU kernels.

## Usage

```python
from nexus import Nexus

# Create tracer
nexus = Nexus(log_level=1)

# Run and get trace
trace = nexus.run(["python", "my_gpu_script.py"])

# Iterate over kernels
for kernel in trace:
    print(f"{kernel.name}: {len(kernel.assembly)} instructions")
    print(kernel.hip)

# Access specific kernel
vector_add = trace["vector_add"]
print(vector_add.assembly)

# Save trace if needed
trace.save("my_trace.json")

# Load old trace
old_trace = Nexus.load("my_trace.json")
```

## Installation

The native C++ library is automatically built when you `pip install`:

```bash
pip install git+https://github.com/AMDResearch/nexus.git
```

That's it! No manual CMake steps needed.

