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

import io
import os
import sys
import json
import glob
import pandas as pd
import selectors
import subprocess
from src.logger import console_error, console_warning, console_debug, console_log

JOIN_TYPE = "grid"

def capture_subprocess_output(subprocess_args: list, new_env=None, profile_mode=False, enable_logging=True) -> tuple:
    console_debug("subprocess", "Running: " + " ".join(subprocess_args))
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = (
        subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        if new_env == None
        else subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            env=new_env,
        )
    )

    # Create callback function for process output
    buf = io.StringIO()

    def handle_output(stream, mask):
        try:
            # Because the process' output is line buffered, there's only ever one
            # line to read when this function is called
            line = stream.readline()
            buf.write(line)
            if enable_logging:
                if profile_mode:
                    console_log("rocprofv3", line.strip(), indent_level=1)
                else:
                    console_log(line.strip())
        except UnicodeDecodeError:
            # Skip this line
            pass

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # If the process terminated, capture any output that remains.
    remaining = process.stdout.read()
    if remaining:
        buf.write(remaining)
        if enable_logging:
            for line in remaining.splitlines():
                console_log(line.strip())
             
    # Get process return code
    return_code = process.wait()
    selector.close()

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)

# Create a dictionary that maps agent ID to agent objects
def get_agent_dict(data):
    agents = data["rocprofiler-sdk-tool"][0]["agents"]

    agent_map = {}

    for agent in agents:
        agent_id = agent["id"]["handle"]
        agent_map[agent_id] = agent

    return agent_map

def v3_json_get_dispatches(data):
    records = data["rocprofiler-sdk-tool"][0]["buffer_records"]

    records_map = {}

    for rec in records["kernel_dispatch"]:
        id = rec["correlation_id"]["internal"]

        records_map[id] = rec

    return records_map

# Returns a dictionary that maps agent ID to GPU ID
# starting at 0.
def get_gpuid_dict(data):

    agents = data["rocprofiler-sdk-tool"][0]["agents"]

    agent_list = []

    # Get agent ID and node_id for GPU agents only
    for agent in agents:

        if agent["type"] == 2:
            agent_id = agent["id"]["handle"]
            node_id = agent["node_id"]
            agent_list.append((agent_id, node_id))

    # Sort by node ID
    agent_list.sort(key=lambda x: x[1])

    # Map agent ID to node id
    map = {}
    gpu_id = 0
    for agent in agent_list:
        map[agent[0]] = gpu_id
        gpu_id = gpu_id + 1

    return map

# Create a dictionary that maps counter ID to counter objects
def v3_json_get_counters(data):
    counters = data["rocprofiler-sdk-tool"][0]["counters"]

    counter_map = {}

    for counter in counters:
        counter_id = counter["id"]["handle"]
        agent_id = counter["agent_id"]["handle"]

        counter_map[(agent_id, counter_id)] = counter

    return counter_map

def v3_json_to_csv(json_file_path, csv_file_path):

    f = open(json_file_path, "rt")
    data = json.load(f)

    dispatch_records = v3_json_get_dispatches(data)
    dispatches = data["rocprofiler-sdk-tool"][0]["callback_records"]["counter_collection"]
    kernel_symbols = data["rocprofiler-sdk-tool"][0]["kernel_symbols"]
    agents = get_agent_dict(data)
    pid = data["rocprofiler-sdk-tool"][0]["metadata"]["pid"]

    gpuid_map = get_gpuid_dict(data)

    counter_info = v3_json_get_counters(data)

    # CSV headers. If there are no dispatches we still end up with a valid CSV file.
    csv_data = dict.fromkeys(
        [
            "Dispatch_ID",
            "GPU_ID",
            "Queue_ID",
            "PID",
            "TID",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Scratch_Per_Workitem",
            "Arch_VGPR",
            "Accum_VGPR",
            "SGPR",
            "Wave_Size",
            "Kernel_Name",
            "Start_Timestamp",
            "End_Timestamp",
            "Correlation_ID",
        ]
    )

    for key in csv_data:
        csv_data[key] = []

    for d in dispatches:

        dispatch_info = d["dispatch_data"]["dispatch_info"]

        agent_id = dispatch_info["agent_id"]["handle"]

        kernel_id = dispatch_info["kernel_id"]

        row = {}

        row["Dispatch_ID"] = dispatch_info["dispatch_id"]

        row["GPU_ID"] = gpuid_map[agent_id]

        row["Queue_ID"] = dispatch_info["queue_id"]["handle"]
        row["PID"] = pid
        row["TID"] = d["thread_id"]

        grid_size = dispatch_info["grid_size"]
        row["Grid_Size"] = grid_size["x"] * grid_size["y"] * grid_size["z"]

        wg = dispatch_info["workgroup_size"]
        row["Workgroup_Size"] = wg["x"] * wg["y"] * wg["z"]

        try:
            row["LDS_Per_Workgroup"] = d["lds_block_size_v"]
        except:
            row["LDS_Per_Workgroup"] = 0

        row["Scratch_Per_Workitem"] = kernel_symbols[kernel_id]["private_segment_size"]
        
        try:
            row["Arch_VGPR"] = d["arch_vgpr_count"]
        except:
            row["Arch_VGPR"] = 0

        # TODO: Accum VGPR is missing from rocprofv3 output.
        row["Accum_VGPR"] = 0

        try:
            row["SGPR"] = d["sgpr_count"]
        except:
            row["SGPR"] = 0

        row["Wave_Size"] = agents[agent_id]["wave_front_size"]

        row["Kernel_Name"] = kernel_symbols[kernel_id]["formatted_kernel_name"]

        id = d["dispatch_data"]["correlation_id"]["internal"]
        rec = dispatch_records[id]

        row["Start_Timestamp"] = rec["start_timestamp"]
        row["End_Timestamp"] = rec["end_timestamp"]
        row["Correlation_ID"] = d["dispatch_data"]["correlation_id"]["external"]

        # Get counters
        ctrs = {}

        records = d["records"]
        for r in records:
            ctr_id = r["counter_id"]["handle"]
            value = r["value"]

            name = counter_info[(agent_id, ctr_id)]["name"]

            if name.endswith("_ACCUM"):
                # It's an accumulate counter. Omniperf expects the accumulated value
                # to be in SQ_ACCUM_PREV_HIRES.
                name = "SQ_ACCUM_PREV_HIRES"

            # Some counters appear multiple times and need to be summed
            if name in ctrs:
                ctrs[name] += value
            else:
                ctrs[name] = value

        # Append counter values
        for ctr, value in ctrs.items():
            row[ctr] = value

        # Add row to CSV data
        for col_name, value in row.items():
            if col_name not in csv_data:
                csv_data[col_name] = []

            csv_data[col_name].append(value)

    df = pd.DataFrame(csv_data)

    df.to_csv(csv_file_path, index=False)

def test_df_column_equality(df):
    return df.eq(df.iloc[:, 0], axis=0).all(1).all()

def join_prof(path):
    """Manually join separated rocprof runs"""
    # Set default output directory if not specified
    out = f"{path}/pmc_perf.csv"
    files = glob.glob(f"{path}/pmc_perf_*.csv")

    df = None
    for i, file in enumerate(files):
        _df = pd.read_csv(file)
        if JOIN_TYPE == "kernel":
            key = _df.groupby("Kernel_Name").cumcount()
            _df["key"] = _df.Kernel_Name + " - " + key.astype(str)
        elif JOIN_TYPE == "grid":
            key = _df.groupby(["Kernel_Name", "Grid_Size"]).cumcount()
            _df["key"] = (
                _df["Kernel_Name"]
                + " - "
                + _df["Grid_Size"].astype(str)
                + " - "
                + key.astype(str)
            )
        else:
            console_error(
                "%s is an unrecognized option for --join-type" % JOIN_TYPE
            )
            sys.exit(1)

        if df is None:
            df = _df
        else:
            # join by unique index of kernel
            df = pd.merge(df, _df, how="inner", on="key", suffixes=("", f"_{i}"))

    # TODO: check for any mismatch in joins
    duplicate_cols = {
        "GPU_ID": [col for col in df.columns if col.startswith("GPU_ID")],
        "Grid_Size": [col for col in df.columns if col.startswith("Grid_Size")],
        "Workgroup_Size": [
            col for col in df.columns if col.startswith("Workgroup_Size")
        ],
        "LDS_Per_Workgroup": [
            col for col in df.columns if col.startswith("LDS_Per_Workgroup")
        ],
        "Scratch_Per_Workitem": [
            col for col in df.columns if col.startswith("Scratch_Per_Workitem")
        ],
        "SGPR": [col for col in df.columns if col.startswith("SGPR")],
    }
    # Check for vgpr counter in ROCm < 5.3
    if "vgpr" in df.columns:
        duplicate_cols["vgpr"] = [col for col in df.columns if col.startswith("vgpr")]
    # Check for vgpr counter in ROCm >= 5.3
    else:
        duplicate_cols["Arch_VGPR"] = [
            col for col in df.columns if col.startswith("Arch_VGPR")
        ]
        duplicate_cols["Accum_VGPR"] = [
            col for col in df.columns if col.startswith("Accum_VGPR")
        ]
    for key, cols in duplicate_cols.items():
        _df = df[cols]
        if not test_df_column_equality(_df):
            msg = "Detected differing {} values while joining pmc_perf.csv".format(
                key
            )
            console_warning(msg)
        else:
            msg = "Successfully joined {} in pmc_perf.csv".format(key)
            console_debug(msg)

    # now, we can:
    #   A) throw away any of the "boring" duplicates
    df = df[
        [
            k
            for k in df.keys()
            if not any(
                k.startswith(check)
                for check in [
                    # rocprofv3 headers
                    "Correlation_ID_",
                    "Wave_Size_",
                    # rocprofv2 headers
                    "GPU_ID_",
                    "Grid_Size_",
                    "Workgroup_Size_",
                    "LDS_Per_Workgroup_",
                    "Scratch_Per_Workitem_",
                    "vgpr_",
                    "Arch_VGPR_",
                    "Accum_VGPR_",
                    "SGPR_",
                    "Dispatch_ID_",
                    "Queue_ID",
                    "Queue_Index",
                    "PID",
                    "TID",
                    "SIG",
                    "OBJ",
                    # rocscope specific merged counters, keep original
                    "dispatch_",
                    # extras
                    "sig",
                    "queue-id",
                    "queue-index",
                    "pid",
                    "tid",
                    "fbar",
                ]
            )
        ]
    ]
    #   B) any timestamps that are _not_ the duration, which is the one we care about
    df = df[
        [
            k
            for k in df.keys()
            if not any(
                check in k
                for check in [
                    "DispatchNs",
                    "CompleteNs",
                    # rocscope specific timestamp
                    "HostDuration",
                ]
            )
        ]
    ]
    #   C) sanity check the name and key
    namekeys = [k for k in df.keys() if "Kernel_Name" in k]
    assert len(namekeys)
    for k in namekeys[1:]:
        assert (df[namekeys[0]] == df[k]).all()
    df = df.drop(columns=namekeys[1:])
    # now take the median of the durations
    bkeys = []
    ekeys = []
    for k in df.keys():
        if "Start_Timestamp" in k:
            bkeys.append(k)
        if "End_Timestamp" in k:
            ekeys.append(k)
    # compute mean begin and end timestamps
    endNs = df[ekeys].mean(axis=1)
    beginNs = df[bkeys].mean(axis=1)
    # and replace
    df = df.drop(columns=bkeys)
    df = df.drop(columns=ekeys)
    df["Start_Timestamp"] = beginNs
    df["End_Timestamp"] = endNs
    # finally, join the drop key
    df = df.drop(columns=["key"])
    # save to file and delete old file(s), skip if we're being called outside of Omniperf
    
    df.to_csv(out, index=False)
    for file in files:
        os.remove(file)
