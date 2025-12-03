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

import os
import re
import sys
import argparse
import json
import glob
import shutil
import shlex
import pandas as pd
from pathlib import Path as path
from src.path import get_project_root
from src.utils import capture_subprocess_output, v3_json_to_csv, join_prof
from src.logger import console_error, console_log, console_warning, console_debug

SUPPORTED_ARCHS = {
    "gfx908": "MI100",
    "gfx90a": "MI200",
    "gfx940": "MI300",
    "gfx941": "MI300",
    "gfx942": "MI300",
    "gfx950": "MI350",
}
arch = ""
# Allow the user to override the ROCm dependencies
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
if not os.path.exists(ROCM_PATH):
    console_error(f"Default ROCm path: \"{ROCM_PATH}\" does not exist. Set ROCM_PATH to resolve.")
ROCPROF_COMPUTE = shutil.which("rocprof-compute")
if not ROCPROF_COMPUTE:
    console_error("rocprof-compute is not detectable via PATH.")
ROCPROFV3 = shutil.which("rocprofv3")
if not ROCPROFV3:
    console_error("rocprofv3 is not detectable via PATH.")

def detect_arch():
    global arch
    # Use rocminfo to detect the architecture
    success, output = capture_subprocess_output(
        [f"{ROCM_PATH}/bin/rocminfo"],
        enable_logging=False,
    )
    if not success:
        console_error("Failed to run rocminfo.")
    
    gfx = None
    for line in output.splitlines():
        if "gfx" in line:
            arch = line.split()
            if len(arch) > 1 and arch[1] in SUPPORTED_ARCHS.keys():
                gfx = arch[1]
                break
            else:
                console_warning(f"Unsupported architecture: {arch[1]}")
    if gfx is None:
        console_error("No supported architecture found.")
    arch = gfx
    console_log(f"Target: {SUPPORTED_ARCHS[arch]}")

def timing_run(app_cmd: list, output_dir: str, top_n: int) -> list:
    # Use --kernel-trace to collect timing data
    success, output = capture_subprocess_output([
        f"{ROCPROFV3}",
        "--kernel-trace",
        "--pmc", "GRBM_GUI_ACTIVE",
        "-o", "/tmp/rocprofv3_timing",
        "--output-format", "json",
        "--"
    ] + app_cmd, profile_mode=True)
    if not success:
        console_error("Failed to run rocprofv3 for timing.")

    output_csv = f"{output_dir}/timing_data.csv"
    v3_json_to_csv("/tmp/rocprofv3_timing_results.json", output_csv)
    # Extract the timing info from the output
    timing_df = pd.read_csv(output_csv)
    # Group by Kernel_Name and calculate total time, average time, count, and percentage
    timing_df['Total_Duration'] = timing_df['End_Timestamp'] - timing_df['Start_Timestamp']
    timing_df['Count'] = timing_df.groupby('Kernel_Name')['Kernel_Name'].transform('size')
    timing_df['Total_Duration'] = timing_df.groupby('Kernel_Name')['Total_Duration'].transform('sum')
    timing_df['Avg_Duration'] = timing_df.groupby('Kernel_Name')['Total_Duration'].transform('mean')
    grouped = timing_df.drop_duplicates(subset=['Kernel_Name']).sort_values(by='Total_Duration', ascending=False)
    grouped['Pct'] = (grouped['Total_Duration'] / grouped['Total_Duration'].sum()) * 100
    grouped.rename(columns={'Total_Duration': 'Total_Duration(ns)', 'Avg_Duration': 'Avg_Duration(ns)'}, inplace=True)
    # Drop the GRBM_GUI_ACTIVE, Dispatch_ID column as its not needed in timing run
    grouped.drop(columns=['GRBM_GUI_ACTIVE', 'Dispatch_ID'], inplace=True)
    grouped = grouped.reset_index(drop=True)
    # Save the timing data to a CSV file
    grouped.to_csv(output_csv, index=False)
    grouped = grouped.head(top_n)["Kernel_Name"].tolist()
    # Update kernel filter based on the top kernels
    console_debug(f"Top {top_n} kernels: {grouped}")
    return grouped

def create_kernel_regex(top_n_kernels: list, output_dir:str):
    global arch
    # Read the input file based on the architecture
    config_subdir = SUPPORTED_ARCHS[arch][:3]
    config_path = os.path.join(get_project_root(), "configs", config_subdir, "input.json")
    with open(config_path, 'r') as f:
        input_json = json.load(f)
    escaped_names = [re.escape(name) for name in top_n_kernels]
    if len(escaped_names) == 0:
        regex_pattern = ".*"
    else:
        regex_pattern = r"^(" + "|".join(escaped_names) + r")$"
    # Add kernel filter to input file and save for submission
    for job in input_json["jobs"]:
        job["kernel_include_regex"] = regex_pattern
    with open(f"{output_dir}/input.json", 'w') as f:
        json.dump(input_json, f, indent=2)

def run_profiler(app_cmd:list, output_dir:str):
    global arch
    # Run the profiler with the updated input file
    success, output = capture_subprocess_output([
        f"{ROCPROFV3}",
        "--input", f"{output_dir}/input.json",
        "--output-directory", str(output_dir),
        "--kernel-trace",
        "--"
    ] + app_cmd, profile_mode=True)
    if not success:
        console_error("Failed to run rocprofv3 for profiling.")

    # Extract the profiling data from the output
    results_files_csv = {}
    results_files_json = glob.glob(f"{output_dir}/*/*.json")
    for json_file in results_files_json:
        csv_file = path(json_file).with_suffix(".csv")
        v3_json_to_csv(json_file, csv_file)
    results_files_csv = glob.glob(f"{output_dir}/*/*.csv")
    for csv_file in results_files_csv:
        shutil.copy(
            csv_file,
            f"{output_dir}/{path(csv_file).name[:-12]}.csv"
        )
        shutil.rmtree(path(csv_file).parent)

    # Combine pmc results into a single CSV file
    join_prof(output_dir)

    console_log(f"Profiling completed successfully. Output saved to \"{output_dir}\".")

def parse_sys_info(output: str) -> dict:
    """
    Parse the system information table from the subprocess output into a dictionary.
    """
    sys_info = {}
    lines = output.splitlines()
    start_parsing = False

    for line in lines:
        if line.startswith("│") :
            columns = [col.strip() for col in line.split("│")[2:-3]]
            # Ensure valid data row
            if (len(columns) >= 2 and columns[0] != "") and columns[0] != "Spec":
                sys_info[columns[0]] = columns[1]

    return sys_info

def collect_sys_info(output_dir: str, workload_name: str, app_cmd: list):
    """
    Collect system information and save it to a JSON file.
    """
    success, output = capture_subprocess_output([
        f"{ROCPROF_COMPUTE}",
        "--specs"
    ], enable_logging=False)
    if not success:
        console_error(f"Failed to collect system information.\n{output}")

    # Parse the system information
    headers = []
    values = []
    lines = output.splitlines()

    for line in lines:
        if line.startswith("│") :
            columns = [col.strip() for col in line.split("│")[2:-3]]
            # Ensure valid data row
            if (len(columns) >= 2 and columns[0] != "") and columns[0] != "Spec":
                headers.append(columns[0])
                values.append(columns[1])
    # Create a DataFrame from the parsed data
    sys_info = pd.DataFrame([values], columns=headers)
    sys_info["Workload Name"] = workload_name
    sys_info["Command"] = " ".join(app_cmd)

    # Save the system information to a csv file
    sys_info.to_csv(f"{output_dir}/sys_info.csv", index=False)

def run_profiling(args) -> str:
    detect_arch()
    output_dir = os.path.join(get_project_root(), "workloads", str(args.name), str(SUPPORTED_ARCHS[arch]))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    app_cmd = shlex.split(args.remaining)

    # Collect required profiling data
    collect_sys_info(output_dir, args.name, app_cmd)
    top_n_kernels = timing_run(app_cmd, output_dir, args.top_n)
    create_kernel_regex(top_n_kernels, output_dir)
    run_profiler(app_cmd, output_dir)
    return str(output_dir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Collect performance data for a ROCm application.")
#     parser.add_argument("remaining", nargs=argparse.REMAINDER, help="Application command to run.")
#     parser.add_argument("--name", type=str, default=time.strftime("%m-%d-%Y-%H-%M"), help="Name of the workload.")
#     parser.add_argument("--top-n", type=int, default=10, help="Number of top kernels to profile.")
#     args = parser.parse_args()

#     if not args.remaining:
#         console_error("No application command provided.")

#     run_profiling(args)
