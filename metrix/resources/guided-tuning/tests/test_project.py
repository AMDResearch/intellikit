##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import subprocess
import pytest
import time
import random
import string
import os

def generate_random_string(length=5):
    """Generate a random string of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

GT = "./bin/gt"
TEST_DB_LOC = f"/tmp/gt_test_{generate_random_string()}.db"
TIMESTAMP = time.strftime("%Y-%m-%d-%H%M")
DB_SIZE_THRESHOLD = 10  # Maximum percent difference allowed

# Test for installation
@pytest.mark.parametrize("dependency", ["python3", "pip", "rocprofv3", "rocprof-compute"])
def test_dependencies_installed(dependency):
    """Ensure required dependencies are installed."""
    assert subprocess.call(["which", dependency]) == 0, f"{dependency} is not installed."

def test_profile_and_load_all():
    """Test the profiling with a sample code."""
    test_binary = "tests/triton/01-vector-add.py"
    result = subprocess.run([
        GT, "profile", 
        "-n", f"triton-test-{TIMESTAMP}-ALL",
        "--db", TEST_DB_LOC,
        "--", "python3", test_binary,
        "-vvv"
    ], capture_output=True)
    assert result.returncode == 0, f"{GT} profile failed."

def test_profile_and_load_top_n():
    """Test the profiling (top-n) with a sample code."""
    test_binary = "tests/triton/01-vector-add.py"
    result = subprocess.run([
        GT, "profile", 
        "-n", f"triton-test-{TIMESTAMP}-TOP-5",
        "--db", TEST_DB_LOC,
        "--top-n", "5",
        "--", "python3", test_binary,
        "-vvv"
    ], capture_output=True)
    assert result.returncode == 0, f"{GT} profile failed."

def test_show_data():
    """Test for listing workloads."""
    result = subprocess.run([
        GT, "db",
        "--db", TEST_DB_LOC,
        "-n", f"triton-test-{TIMESTAMP}-ALL",
        "-vvv"
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"{GT} db failed."
    assert f"triton-test-{TIMESTAMP}-ALL" in result.stdout, f"Cannot find test workload in DB."

def test_single_workload_query():
    """Test inspecting a single workload by ID."""
    result = subprocess.run([
        GT, "db",
        "--db", TEST_DB_LOC,
        "-w", "1",
        "-vvv"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Failed to inspect workload."

def test_dispatch_filter_query():
    """Test filtering with a dispatch index."""
    dispatch_index = "4"
    result = subprocess.run([
        GT, "db",
        "--db", TEST_DB_LOC,
        "-w", "1", 
        "-d", dispatch_index,
        "-vvv"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Dispatch filtering failed."

def test_kernel_filter_query():
    """Test filtering kernels using regex."""
    kernel_filter = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"
    result = subprocess.run([
        GT, "db",
        "--db", TEST_DB_LOC,
        "-w", "1", 
        "-k", kernel_filter,
        "-vvv"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Kernel filtering failed."

def test_dispatch_kernel_query():
    """Test the kernel with dispatch filtering functionality."""
    workload_id = "1"
    dispatch_id = "4"
    result = subprocess.run([
        GT, "db",
        "--db", TEST_DB_LOC,
        "-w", workload_id, 
        "-d", dispatch_id,
        "-vvv"
    ], capture_output=True, text=True)
    assert result.returncode == 0, "Dispatch+Kernel filtering failed."

def test_db_size():
    """Test that the size of new DB does not differ significantly from the baseline."""
    baseline_path = "tests/BASELINE.db"
    assert os.path.exists(baseline_path), "Baseline database file does not exist."
    assert os.path.exists(TEST_DB_LOC), "Test database file does not exist."

    baseline_size = os.path.getsize(baseline_path)
    test_db_size = os.path.getsize(TEST_DB_LOC)

    size_difference = abs(test_db_size - baseline_size) / baseline_size * 100
    assert size_difference <= DB_SIZE_THRESHOLD, (
        f"Database size difference ({size_difference:.2f}%) exceeds threshold of {DB_SIZE_THRESHOLD}%."
    )

if __name__ == "__main__":
    pytest.main()