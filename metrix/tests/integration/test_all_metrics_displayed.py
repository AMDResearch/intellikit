"""
Integration test to verify all metrics are displayed

This test validates that all memory and compute metrics are properly
computed and displayed by the metrix profiler.
"""

import pytest
import subprocess
from pathlib import Path


@pytest.fixture
def vector_add_binary(tmp_path):
    """Compile vector_add kernel"""
    source = Path(__file__).parent.parent.parent / "examples" / "01_vector_add" / "kernel.hip"
    binary = tmp_path / "vector_add"

    result = subprocess.run(
        ["hipcc", str(source), "-o", str(binary)],
        capture_output=True,
        timeout=30,
    )

    if result.returncode != 0:
        pytest.skip(f"Could not compile kernel: {result.stderr.decode()}")

    return binary


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_all_memory_metrics_are_displayed(vector_add_binary):
    """Verify that all 12 memory metrics are computed and displayed"""
    result = subprocess.run(
        ["metrix", "-n", "1", "--aggregate", "--profile", "memory", str(vector_add_binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = result.stdout

    # List of all expected memory metrics (friendly names as displayed)
    expected_memory_metrics = [
        # Memory Bandwidth (5 metrics)
        "HBM Read Bandwidth",
        "HBM Write Bandwidth",
        "HBM Bandwidth Utilization",
        "L2 Cache Bandwidth Utilization",
        "Total HBM Bytes Transferred",
        # Cache Performance (2 metrics)
        "L2 Cache Hit Rate",
        "L1 Cache Hit Rate",
        # Memory Access Patterns (3 metrics)
        "Memory Coalescing Efficiency",
        "Global Load Efficiency",
        "Global Store Efficiency",
        # Local Data Share (1 metric)
        "LDS Bank Conflicts",
    ]

    missing_metrics = []
    for metric in expected_memory_metrics:
        if metric not in output:
            missing_metrics.append(metric)

    assert len(missing_metrics) == 0, (
        f"Missing memory metrics: {missing_metrics}\n\nOutput:\n{output}"
    )

    print(f"✓ All {len(expected_memory_metrics)} memory metrics displayed successfully")


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_bandwidth_metrics_have_values(vector_add_binary):
    """Verify bandwidth metrics compute to non-zero values"""
    result = subprocess.run(
        ["metrix", "-n", "1", "--aggregate", str(vector_add_binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = result.stdout

    # These metrics were previously computing to 0.00 due to the bug
    # Now they should have real values
    assert "HBM Read Bandwidth" in output
    assert "HBM Write Bandwidth" in output

    # Extract the bandwidth values and verify they're non-zero
    lines = output.split("\n")
    for line in lines:
        if "HBM Read Bandwidth" in line:
            # Format: "  HBM Read Bandwidth                     XX.XX GB/s"
            assert "0.00 GB/s" not in line, "HBM Read Bandwidth is zero!"
        if "HBM Write Bandwidth" in line:
            assert "0.00 GB/s" not in line, "HBM Write Bandwidth is zero!"
        if "HBM Bandwidth Utilization" in line:
            assert "0.00 percent" not in line, "HBM Bandwidth Utilization is zero!"


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_all_compute_metrics_are_displayed(vector_add_binary):
    """Verify that all compute metrics are computed and displayed"""
    result = subprocess.run(
        ["metrix", "-n", "1", "--aggregate", "--profile", "compute", str(vector_add_binary)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = result.stdout

    # List of all expected compute metrics (friendly names as displayed)
    expected_compute_metrics = [
        "Total FLOPS",
        "HBM Compute Throughput",
        "HBM Arithmetic Intensity",
        "L2 Arithmetic Intensity",
        "L1 Arithmetic Intensity",
    ]

    missing_metrics = []
    for metric in expected_compute_metrics:
        if metric not in output:
            missing_metrics.append(metric)

    assert len(missing_metrics) == 0, (
        f"Missing compute metrics: {missing_metrics}\n\nOutput:\n{output}"
    )

    print(f"✓ All {len(expected_compute_metrics)} compute metrics displayed successfully")


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_json_output_has_memory_metrics(vector_add_binary, tmp_path):
    """Verify JSON output contains all memory metrics"""
    output_file = tmp_path / "results.json"

    result = subprocess.run(
        [
            "metrix",
            "-n",
            "1",
            "--aggregate",
            "--profile",
            "memory",
            "-o",
            str(output_file),
            str(vector_add_binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output_file.exists()

    import json

    with open(output_file) as f:
        data = json.load(f)

    # Check structure (new format: dispatch_key -> {duration_us, metrics})
    assert len(data) > 0, "No kernels in JSON output"

    # Get first kernel/dispatch
    first_key = list(data.keys())[0]
    kernel_data = data[first_key]

    assert "duration_us" in kernel_data
    assert "metrics" in kernel_data

    # Verify key memory bandwidth metrics
    assert "memory.hbm_bandwidth_utilization" in kernel_data["metrics"]
    assert "memory.hbm_read_bandwidth" in kernel_data["metrics"]
    assert "memory.hbm_write_bandwidth" in kernel_data["metrics"]
    assert "memory.l2_bandwidth" in kernel_data["metrics"]


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_json_output_has_compute_metrics(vector_add_binary, tmp_path):
    """Verify JSON output contains all compute metrics"""
    output_file = tmp_path / "results.json"

    result = subprocess.run(
        [
            "metrix",
            "-n",
            "1",
            "--aggregate",
            "--profile",
            "compute",
            "-o",
            str(output_file),
            str(vector_add_binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output_file.exists()

    import json

    with open(output_file) as f:
        data = json.load(f)

    # Check structure
    assert len(data) > 0, "No kernels in JSON output"

    # Get first kernel/dispatch
    first_key = list(data.keys())[0]
    kernel_data = data[first_key]

    assert "duration_us" in kernel_data
    assert "metrics" in kernel_data

    # Verify compute metrics are present
    assert "compute.total_flops" in kernel_data["metrics"]
    assert "compute.hbm_gflops" in kernel_data["metrics"]
    assert "compute.hbm_arithmetic_intensity" in kernel_data["metrics"]
    assert "compute.l2_arithmetic_intensity" in kernel_data["metrics"]
    assert "compute.l1_arithmetic_intensity" in kernel_data["metrics"]
