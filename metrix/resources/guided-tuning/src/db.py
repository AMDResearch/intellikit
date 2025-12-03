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

import duckdb
from src.path import get_data_path

DB_FILE = get_data_path() + '/guided.db'
def get_db_file(): return DB_FILE
def set_db_file(file):
  global DB_FILE
  DB_FILE = file

def set_db_file(file):
  global DB_FILE
  DB_FILE = file

def get_db_file():
   return DB_FILE

def sql(sql_statement):
    CON = duckdb.connect(get_db_file())
    results = None
    try:
      results = CON.sql(sql_statement).fetchall()
    except AttributeError:
      pass
    return results

def remove(workload):
  _ = sql(f'delete from workload where workload_id={workload}')
  _ = sql(f'delete from top where workload={workload}')
  _ = sql(f'delete from dispatch where workload={workload}')
  _ = sql(f'delete from metric where workload={workload}')

def init():
    con = duckdb.connect(get_db_file())
    con.sql('CREATE SEQUENCE id_sequence START 1;')
    con.sql('''CREATE TABLE workload (
        workload_id INTEGER DEFAULT nextval('id_sequence'),
        workload_name VARCHAR,
        command VARCHAR,
        ip_block VARCHAR,
        timestamp VARCHAR,
        hostname VARCHAR,
        cpu_model VARCHAR,
        sbios VARCHAR,
        linux_distro VARCHAR,
        linux_kernel_version VARCHAR,
        amd_gpu_kernel_version VARCHAR,
        cpu_memory BIGINT,
        gpu_memory BIGINT,
        rocm_version VARCHAR,
        vbios VARCHAR,
        compute_partition VARCHAR,
        memory_partition VARCHAR,
        gpu_series VARCHAR,
        gpu_model VARCHAR,
        gpu_arch VARCHAR,
        chip_id INTEGER,
        gpu_l1 INTEGER,
        gpu_l2 INTEGER,
        cu_per_gpu INTEGER,
        simd_per_cu INTEGER,
        se_per_gpu INTEGER,
        wave_size INTEGER,
        workgroup_max_size INTEGER,
        max_waves_per_cu INTEGER,
        max_sclk INTEGER,
        max_mclk INTEGER,
        cur_sclk INTEGER,
        cur_mclk INTEGER,
        total_l2_chan INTEGER,
        lds_banks_per_cu INTEGER,
        sqc_per_gpu INTEGER,
        pipes_per_gpu INTEGER,
        num_xcd INTEGER,
        hbm_channels DOUBLE
    );''')
    con.sql('''CREATE TABLE dispatch (
        workload INTEGER,
        dispatch INTEGER,
        kernel VARCHAR,
        gpu INTEGER,
        grid_size INTEGER,
        workgroup_size INTEGER,
        lds_per_workgroup INTEGER,
        scratch_per_workitem INTEGER,
        arch_vgpr INTEGER,
        accum_vgpr INTEGER,
        sgpr INTEGER,
        wave_size INTEGER,
        start_ns BIGINT,
        end_ns BIGINT,
        correlation_id BIGINT,
    );''')
    con.sql('''CREATE TABLE top (
        workload INTEGER,
        kernel VARCHAR,
        count INTEGER,
        duration_sum DOUBLE,
        duration_mean DOUBLE,
        pct DOUBLE
    );''')
    con.sql('''CREATE TABLE metric (
        workload INTEGER,
        dispatch INTEGER,
        name VARCHAR,
        value BIGINT
    );''')
    con.close()