# Analyze Mode


.. warning::
   **⚠** The `analyze` mode is still under active development. Features and outputs may change, and some functionality may not yet be fully implemented or stable.


## Overview

The `analyze` mode is used to analyze database results for specific workloads and dispatches. It provides insights into GPU performance characteristics, occupancy, and potential bottlenecks. This mode is designed to help users identify areas for optimization in their workloads.

## Usage

```bash
gt analyze [options]
```

### Command Line Options

| Option                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `workload_id`           | **(Required)** The workload ID to analyze.                                 |
| `dispatch_id`           | **(Required)** The dispatch ID to analyze.                                 |
| `-s`, `--silence`       | Suppress output messages and run in silent mode.                           |

### Example Usage

```bash
# Analyze workload ID 1 and dispatch ID 4
gt analyze 1 4

# Analyze workload ID 2 and dispatch ID 3 in silent mode
gt analyze 2 3 --silence
```

## Key Features

- **Occupancy Analysis:** Provides insights into expected and achieved occupancy, highlighting potential resource bottlenecks such as VGPR, SGPR, LDS, or wavefront limits.
- **Memory Access Patterns:** Evaluates memory access patterns and suggests optimizations for improving coalesced memory access.
- **Compute vs. Memory Bound:** Identifies whether the workload is compute-bound or memory-bound and provides recommendations for optimization.
- **Detailed Recommendations:** Offers actionable suggestions to improve performance based on the analysis.

## Output

The `analyze` mode generates a detailed report that includes:

1. **Main Characteristics:** Key metrics such as compute/memory ratio, L1 data per wave, and VMEM instructions per wave.
2. **Occupancy Analysis:** Expected vs. achieved occupancy, along with potential bottlenecks and improvement suggestions.
3. **Memory Access Recommendations:** Suggestions for optimizing memory access patterns to improve performance.

## Notes

- Ensure that the database contains valid profiling data for the specified workload and dispatch IDs before running the `analyze` mode.
- The `analyze` mode relies on metrics collected during profiling. Incomplete or inaccurate profiling data may affect the quality of the analysis.

## Limitations

- The `analyze` mode currently does not support advanced visualization or export of analysis results.
- Some recommendations may require additional tools (e.g., SQTT) or manual instrumentation for further investigation.