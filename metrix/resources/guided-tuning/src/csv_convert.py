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

import os, sys
import pandas as pd
import duckdb
from src.db import init, get_db_file, set_db_file
from src.metric_names import METRICS
from src.logger import console_error, console_log, console_debug
import tempfile

def check_path(path):
    if os.path.exists(f'{path}/sys_info.csv'):
        return [path]
    else:
        return [path + '/' + p for p in os.listdir(path)
                if os.path.exists(f'{path}/{p}/sys_info.csv')]

def load_workload(workload):
    con = duckdb.connect(get_db_file())
    curr_metric_count = 0
    try:
        curr_metric = con.sql('select count(*) from metric').fetchall()
        curr_metric_count = curr_metric[0][0]
    except duckdb.duckdb.CatalogException:
        pass

    # update workload
    results = con.sql("SELECT nextval('id_sequence') as nextval").fetchall()
    db_workload_id = results[0][0]
    sys_info = pd.read_csv(f'{workload}/sys_info.csv')

    sys_info.insert(0, 'workload_id', [db_workload_id], True)
    try:
        con.sql("INSERT INTO workload SELECT * FROM sys_info")
    except duckdb.duckdb.BinderException as e:
        msg = f"Failed to insert sys_info into table due to a mismatch in expected columns.\n" \
        f"Please check the sys_info.csv file in \"{workload}\" directory against \"prep/db.py\".\n" \
        f"{e}"
        console_error(msg)
        
    # update dispatch
    dispatches = pd.read_csv(f'{workload}/pmc_perf.csv')
    columns = {
        'Dispatch_ID': 'dispatch',
        'Kernel_Name': 'kernel',
        'GPU_ID': 'gpu',
        'Grid_Size': 'grid_size',
        'Workgroup_Size': 'workgroup_size',
        'Wave_Size': 'wave_size',
        'LDS_Per_Workgroup': 'lds_per_workgroup',
        'Scratch_Per_Workitem': 'scratch_per_workitem',
        'Arch_VGPR': 'arch_vgpr',
        'Accum_VGPR': 'accum_vgpr',
        'SGPR': 'sgpr',
        'Start_Timestamp': 'start_ns',
        'End_Timestamp': 'end_ns',
        'Correlation_ID': 'correlation_id',
    }
    dispatches = dispatches[list(columns.keys())]
    dispatches.rename(columns=columns, inplace=True)
    dispatches.insert(0, 'workload', [db_workload_id] * dispatches.shape[0], True)
    con.sql("INSERT INTO dispatch SELECT * FROM dispatches")

    # update top
    # assume the following has been run:
    #   rocprof-compute analyze --path workloads/name/MI200 --list-stats
    try:
        top_kernels = pd.read_csv(f'{workload}/timing_data.csv')
        top_kernels.rename(columns={'Kernel_Name': 'kernel',
                            'Count': 'count',
                            'Total_Duration(ns)': 'duration_sum',
                            'Avg_Duration(ns)': 'duration_mean',
                            'Pct': 'pct'},
                inplace=True)
        top_kernels = top_kernels[['kernel', 'count', 'duration_sum', 'duration_mean', 'pct']]
        top_kernels.insert(0, 'workload', [db_workload_id] * top_kernels.shape[0], True)
        con.sql("INSERT INTO top SELECT * FROM top_kernels")
    except FileNotFoundError:
        console_error(f'Please run the command first:\n\trocprof-compute analyze --path {workload} --list-stats')

    # update metric
    temp = tempfile.NamedTemporaryFile(suffix='.guided')
    temp.write(b'workload|dispatch|name|value\n')

    load_perf_datafile(db_workload_id, workload, 'pmc_perf.csv', temp)
    load_perf_datafile(db_workload_id, workload, 'SQ_INSTS_LDS.csv', temp)
    load_perf_datafile(db_workload_id, workload, 'SQ_INSTS_VMEM.csv', temp)
    load_perf_datafile(db_workload_id, workload, 'SQ_INSTS_SMEM.csv', temp)
    load_perf_datafile(db_workload_id, workload, 'SQ_WAVE_CYCLES.csv', temp)

    temp.seek(0);
    temp_to_load = pd.read_table(temp.name, sep='|')

    con.sql("INSERT INTO metric SELECT * FROM temp_to_load")

    temp.close()

    metric_count = con.sql('select count(*) from metric').fetchall()
    metric_count = metric_count[0][0]
    console_debug(f'{metric_count - curr_metric_count:,d} metrics loaded.')

    con.close()

def export_db(df:pd.DataFrame, filename:str):
    df.to_csv(filename, index=False)


METRIC_RENAME = {
    'SQ_INSTS_SMEM.csv': {
        'SQ_ACCUM_PREV_HIRES': 'SQ_ACCUM_PREV_HIRES_SMEM'
    },
    'SQ_WAVE_CYCLES.csv': {
        'SQ_ACCUM_PREV_HIRES': 'SQ_ACCUM_PREV_HIRES_WAVES'
    },
    'SQ_INSTS_LDS.csv': {
        'SQ_ACCUM_PREV_HIRES': 'SQ_ACCUM_PREV_HIRES_LDS'
    },
    'SQ_INSTS_VMEM.csv': {
        'SQ_ACCUM_PREV_HIRES': 'SQ_ACCUM_PREV_HIRES_VMEM'
    }
}

def load_perf_datafile(workload_id, workload, datafile, temp):
    console_log(f'\tLoading {workload}/{datafile} ...')
    perf = pd.read_csv(f'{workload}/{datafile}')
    if datafile in METRIC_RENAME:
        perf.rename(columns=METRIC_RENAME[datafile], inplace=True)
    perf = perf.to_dict('records')
    for x in perf:
        dispatch_id = x['Dispatch_ID']
        for k in x.keys():
            if k in METRICS:
                row = f'{workload_id}|{dispatch_id}|{k}|{int(x[k])}\n'
                temp.write(row.encode())

def run_convert(csv_dir:str, db_file:str=None):
    if db_file:
        set_db_file(db_file)

    if not os.path.exists(get_db_file()):
        console_log('Initializing Database...')
        init()

    for workload in check_path(csv_dir):
        console_log(f'Loading workload({workload}) ...')
        load_workload(workload)

# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         console_error('Usage: python3 csv_convert.py <csv_dir> [db_file]')

#     csv_dir = sys.argv[1]
#     db_file = None
#     if len(sys.argv) > 2:
#         db_file = sys.argv[2]

#     run_convert(csv_dir, db_file)