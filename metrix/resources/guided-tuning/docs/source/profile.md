# Profiling Mode

The `profile` mode in the Guided Tuning (GT) tool is designed to collect performance data with help from the ROCm Profiler. This mode leverages the `collect_perfmon.py` script to gather detailed profiling information, including kernel execution times, system information, and performance metrics.

## Command Line Usage

The `profile` mode can be invoked using the following command:

```bash
gt profile [options] -- <application-binary>
```

### Options

| Option         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `--top-n`      | Limit profiling to the top N kernels based on execution time.              |
| `-n`           | Assign a custom name to the workload for database entry.                   |
| `--db`         | Specify the path to the output database file.                              |
| `--`    | Provide the application binary and its arguments after the `--` separator. |

### Example Usage

```bash
gt profile --top-n 5 -n docs-sample -- ./my_app --arg1 --arg2
```

This command profiles the application `my_app` with arguments `--arg1` and `--arg2`, limits profiling to the top 5 kernels, and assigns the workload the name `docs-sample`.

---

## Profiling Workflow

The profiling process involves the following steps:

1. **Architecture Detection**  
    The script uses `rocminfo` to detect the GPU architecture and ensure compatibility with supported architectures:
    - `gfx908` (MI100)
    - `gfx90a` (MI200)
    - `gfx940`, `gfx941`, `gfx942` (MI300)
    - `gfx950` (MI350)

    If the architecture is unsupported, the process terminates with an error.

2. **System Information Collection**  

    The [`rocprof-compute`](https://github.com/ROCm/rocprofiler-compute) tool is used to gather system specifications, which are saved in a `sys_info.csv` file.

3. **Timing Run** (Optional)

    The [`rocprofv3`]((https://github.com/ROCm/rocprofiler-compute)) tool performs a timing run to identify the top N kernels based on execution time. The results are saved in a `timing_data.csv` file.

4. **Kernel Filtering**  

    A kernel filter is created using the top N kernels and added to the profiling configuration file (`input.json`).

5. **Profiling Execution**  

    The [`rocprofv3`](https://github.com/ROCm/rocprofiler-compute) tool is run with the updated configuration to collect detailed profiling data. The results are converted to CSV format and saved in the output directory.

---

## Output

The profiling results are stored in the following structure:

```
workloads/
└── <workload_name>/
     └── <architecture>/
          ├── sys_info.csv
          ├── timing_data.csv
          ├── input.json
          ├── SQ_*.csv
          └── pmc_perf.csv
```

- **`sys_info.csv`**: Contains system specifications and workload metadata.
- **`timing_data.csv`**: Lists kernel execution times and statistics.
- **`input.json`**: Configuration file with kernel filters.
- **`SQ_*.csv`** and **`pmc_perf.csv`**: Detailed profiling data for each kernel, including performance metrics. Separated for readability.

---

For more details, refer to the source code in `collect_perfmon.py`.  